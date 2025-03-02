import asyncio
import json
import logging
import os
import pickle
import shutil
import stat
import subprocess
import time
from argparse import Namespace
from base64 import b64encode
from random import Random
from statistics import median, mean
from typing import Dict, List, Optional, Tuple

import torch

import yappi

import numpy as np

from dlsim.core.model_manager import ModelManager
from dlsim.core.model_evaluator import ModelEvaluator
from dlsim.core.session_settings import SessionSettings
from dlsim.plexus.community import PlexusCommunity
from dlsim.dl.community import DLCommunity
from dlsim.gl.community import GLCommunity

from ipv8.configuration import ConfigBuilder
from ipv8.taskmanager import TaskManager
from ipv8_service import IPv8

from simulation.discrete_loop import DiscreteLoop
from simulation.simulation_endpoint import SimulationEndpoint

from simulations.dl.bypass_network_community import DLBypassNetworkCommunity
from simulations.plexus.bypass_network_community import PlexusBypassNetworkCommunity
from simulations.gl.bypass_network_community import GLBypassNetworkCommunity
from simulations.logger import SimulationLoggerAdapter


class LearningSimulation(TaskManager):
    """
    Base class for any simulation that involves learning.
    """

    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.args = args
        self.session_settings: Optional[SessionSettings] = None
        self.nodes = []
        self.data_dir = os.path.join("data", "n_%d_%s_sd%d" % (self.args.peers, self.args.dataset, self.args.seed))
        self.evaluator = None
        self.logger = None
        self.model_manager: Optional[ModelManager] = None

        self.loop = DiscreteLoop()
        asyncio.set_event_loop(self.loop)

    def get_ipv8_builder(self, peer_id: int) -> ConfigBuilder:
        builder = ConfigBuilder().clear_keys().clear_overlays()

        key_str = chr(peer_id).encode() * 1000
        key_base = b"LibNaCLSK:%s" % key_str[:68]
        key_material = b64encode(key_base).decode()
        builder.add_key_from_bin("my peer", key_material, file_path=os.path.join(self.data_dir, f"ec{peer_id}.pem"))
        return builder

    async def start_ipv8_nodes(self) -> None:
        for peer_id in range(1, self.args.peers + 1):
            if peer_id % 100 == 0:
                print("Created %d peers..." % peer_id)  # The logger has not been setup at this point
            endpoint = SimulationEndpoint()
            instance = IPv8(self.get_ipv8_builder(peer_id).finalize(), endpoint_override=endpoint,
                            extra_communities={
                                'DLCommunity': DLCommunity,
                                'DLBypassNetworkCommunity': DLBypassNetworkCommunity,
                                'PlexusCommunity': PlexusCommunity,
                                'PlexusBypassNetworkCommunity': PlexusBypassNetworkCommunity,
                                'GLCommunity': GLCommunity,
                                'GLBypassNetworkCommunity': GLBypassNetworkCommunity,
                            })
            await instance.start()

            # Set the WAN address of the peer to the address of the endpoint
            for overlay in instance.overlays:
                overlay.max_peers = -1
                overlay.my_peer.address = instance.overlays[0].endpoint.wan_address
                overlay.my_estimated_wan = instance.overlays[0].endpoint.wan_address
                overlay.cancel_pending_task("_check_tasks")  # To ignore the warning for long-running tasks
                overlay.logger = SimulationLoggerAdapter(overlay.logger, {})
                overlay.peer_manager.logger = SimulationLoggerAdapter(overlay.peer_manager.logger, {})
                if self.args.bypass_model_transfers:
                    overlay.bw_scheduler.logger = SimulationLoggerAdapter(overlay.peer_manager.logger, {})

            self.nodes.append(instance)

    def setup_directories(self) -> None:
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)
        os.makedirs(self.data_dir, exist_ok=True)

    def setup_logger(self) -> None:
        root = logging.getLogger()
        root.handlers[0].setFormatter(logging.Formatter("%(asctime)s:%(levelname)s:%(message)s"))
        root.setLevel(getattr(logging, self.args.log_level))

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger = SimulationLoggerAdapter(self.logger, {})

    def ipv8_discover_peers(self) -> None:
        peers_list = [node.overlays[0].my_peer for node in self.nodes]
        for node in self.nodes:
            node.overlays[0].peers_list = peers_list

    def apply_availability_traces(self):
        if not self.args.availability_traces:
            return

        self.logger.info("Applying availability trace file %s", self.args.availability_traces)
        with open(self.args.availability_traces, "rb") as traces_file:
            data = pickle.load(traces_file)

        rand = Random(self.args.seed)
        device_ids = rand.sample(list(data.keys()), self.args.peers)
        for ind, node in enumerate(self.nodes):
            if node.overlays[0].my_peer.public_key.key_to_bin() != self.session_settings.dfl.fixed_aggregator:
                node.overlays[0].set_traces(data[device_ids[ind]])
            else:
                self.logger.error("Not applying availability traces to server node %d", ind)

    def apply_fedscale_traces(self):
        self.logger.info("Applying capability trace file %s", self.args.availability_traces)
        with open(os.path.join("data", "fedscale_traces"), "rb") as traces_file:
            data = pickle.load(traces_file)

        rand = Random(self.args.seed)
        device_ids = rand.sample(list(data.keys()), self.args.peers)
        nodes_bws: Dict[bytes, int] = {}
        for ind, node in enumerate(self.nodes):
            node.overlays[0].model_manager.model_trainer.simulated_speed = data[device_ids[ind]]["computation"]
            if self.args.bypass_model_transfers:
                # Also apply the network latencies
                if self.session_settings.dfl is not None and node.overlays[0].my_peer.public_key.key_to_bin() == self.session_settings.dfl.fixed_aggregator:
                    self.logger.error("Setting BW limit of server node %d to unlimited", ind)
                    bw_limit: int = 1000000000000
                else:
                    bw_limit: int = int(data[ind + 1]["communication"]) * 1024 // 8
                node.overlays[0].bw_scheduler.bw_limit = bw_limit
                nodes_bws[node.overlays[0].my_peer.public_key.key_to_bin()] = bw_limit

        for node in self.nodes:
            node.overlays[0].other_nodes_bws = nodes_bws

    def apply_diablo_traces(self):
        # Read and process the latency matrix
        bw_means = []
        with open(os.path.join("data", "diablo.txt"), "r") as diablo_file:
            rows = diablo_file.readlines()
            for row in rows:
                values = list(map(float, row.strip().split(',')))
                mean_value = np.mean(values) * 1024 * 1024 // 8
                bw_means.append(mean_value)

        nodes_bws: Dict[bytes, int] = {}
        for ind, node in enumerate(self.nodes):
            # TODO this is rather arbitrary for now
            node.overlays[0].model_manager.model_trainer.simulated_speed = 100
            bw_limit: int = bw_means[ind % len(bw_means)]
            node.overlays[0].bw_scheduler.bw_limit = bw_limit
            nodes_bws[node.overlays[0].my_peer.public_key.key_to_bin()] = bw_limit

        for node in self.nodes:
            node.overlays[0].other_nodes_bws = nodes_bws

    def apply_compute_and_bandwidth_traces(self):
        if self.args.traces == "none":
            return
        elif self.args.traces == "fedscale":
            self.apply_fedscale_traces()
        elif self.args.traces == "diablo":
            self.apply_diablo_traces()
        else:
            raise RuntimeError("Unknown traces %s" % self.args.traces)

    def apply_traces(self):
        """
        Set the relevant traces.
        """
        self.apply_availability_traces()
        self.apply_compute_and_bandwidth_traces()

        # Log these bandwidths
        with open(os.path.join(self.data_dir, "bandwidths.csv"), "w") as out_file:
            out_file.write("bandwidth\n")
            for node in self.nodes:
                out_file.write("%d\n" % node.overlays[0].bw_scheduler.bw_limit)

        self.logger.info("Traces applied!")

    def apply_latencies(self):
        """
        If specified in the settings, add latencies between the endpoints.
        """
        if not self.args.latencies_file:
            return

        latencies = []
        with open(self.args.latencies_file) as latencies_file:
            for line in latencies_file.readlines():
                latencies.append([float(l) for l in line.strip().split(",")])

        self.logger.info("Read latency matrix with %d sites!" % len(latencies))

        # Assign nodes to sites in a round-robin fashion and apply latencies accordingly
        for from_ind, from_node in enumerate(self.nodes):
            for to_ind, to_node in enumerate(self.nodes):
                from_site_ind = from_ind % len(latencies)
                to_site_ind = to_ind % len(latencies)
                latency_ms = int(latencies[from_site_ind][to_site_ind]) / 1000
                from_node.endpoint.latencies[to_node.endpoint.wan_address] = latency_ms

        self.logger.info("Latencies applied!")

    def determine_peer_with_lowest_median_latency(self, eligible_peers: List[int]) -> int:
        """
        Based on the latencies, determine the ID of the peer with the lowest median latency to other peers.
        """
        latencies = []
        with open(self.args.latencies_file) as latencies_file:
            for line in latencies_file.readlines():
                latencies.append([float(l) for l in line.strip().split(",")])

        lowest_median_latency = 100000
        lowest_peer_id = 0
        avg_latencies = []
        for peer_id in range(min(len(self.nodes), len(latencies))):
            if peer_id not in eligible_peers:
                continue
            median_latency = median(latencies[peer_id])
            avg_latencies.append(mean(latencies[peer_id]))
            if median_latency < lowest_median_latency:
                lowest_median_latency = median_latency
                lowest_peer_id = peer_id

        self.logger.info("Determined peer %d with lowest median latency: %f", lowest_peer_id + 1, lowest_median_latency)
        self.logger.debug("Average latency: %f" % mean(avg_latencies))
        return lowest_peer_id

    async def setup_simulation(self) -> None:
        self.logger.info("Setting up simulation with %d peers..." % self.args.peers)
        with open(os.path.join(self.data_dir, "accuracies.csv"), "w") as out_file:
            out_file.write("dataset,seed,learning_rate,group,time,peer,round,accuracy,loss\n")

        if self.args.activity_log_interval:
            with open(os.path.join(self.data_dir, "activities.csv"), "w") as out_file:
                out_file.write("time,online,offline,min_nodes_in_view,max_nodes_in_view,avg_nodes_in_view,median_nodes_in_view\n")
            self.register_task("check_activity", self.check_activity, interval=self.args.activity_log_interval)

        if self.args.flush_statistics_interval:
            self.register_task("flush_statistics", self.flush_statistics, interval=self.args.flush_statistics_interval)

        if self.args.bypass_model_transfers:
            with open(os.path.join(self.data_dir, "transfers.csv"), "w") as out_file:
                out_file.write("from,to,round,start_time,duration,type,success\n")

    def check_activity(self):
        """
        Count the number of online/offline peers and write it away.
        """
        online, offline = 0, 0
        active_nodes_in_view: List[int] = []
        for node in self.nodes:
            if node.overlays[0].is_active:
                online += 1
                active_nodes_in_view.append(len(node.overlays[0].peer_manager.get_active_peers()))
            else:
                offline += 1

        cur_time = asyncio.get_event_loop().time()
        with open(os.path.join(self.data_dir, "activities.csv"), "a") as out_file:
            out_file.write("%d,%d,%d,%d,%d,%f,%f\n" % (
                cur_time, online, offline, min(active_nodes_in_view), max(active_nodes_in_view),
                sum(active_nodes_in_view) / len(active_nodes_in_view), median(active_nodes_in_view)))

    async def start_simulation(self) -> None:
        active_nodes: List = []
        for ind, node in enumerate(self.nodes):
            if not node.overlays[0].traces or (node.overlays[0].traces and node.overlays[0].traces["active"][0] == 0):
                node.overlays[0].start()
                active_nodes.append(node)
        self.logger.info("Started %d nodes...", len(active_nodes))

        self.start_nodes_training(active_nodes)

        dataset_base_path: str = self.args.dataset_base_path or os.environ["HOME"]
        if self.args.dataset in ["cifar10", "mnist", "google_speech"]:
            data_dir = os.path.join(dataset_base_path, "dfl-data")
        else:
            # The LEAF dataset
            data_dir = os.path.join(dataset_base_path, "leaf", self.args.dataset)

        if not self.args.bypass_training:
            self.evaluator = ModelEvaluator(data_dir, self.session_settings)

        if self.args.profile:
            yappi.start(builtins=True)

        start_time = time.time()
        if self.args.duration > 0:
            await asyncio.sleep(self.args.duration)
            self.logger.info("Simulation took %f seconds" % (time.time() - start_time))
            self.on_simulation_finished()
            self.loop.stop()
        else:
            self.logger.info("Running simulation for undefined time")

    def start_nodes_training(self, active_nodes: List) -> None:
        pass

    def on_ipv8_ready(self) -> None:
        """
        This method is called when IPv8 is started and peer discovery is finished.
        """
        pass

    def checkpoint_models(self, round_nr: int):
        """
        Dump all models during a particular round.
        """
        models_dir = os.path.join(self.data_dir, "models", "%d" % round_nr)
        shutil.rmtree(models_dir, ignore_errors=True)
        os.makedirs(models_dir, exist_ok=True)

        avg_model = self.model_manager.aggregate_trained_models()
        for peer_ind, node in enumerate(self.nodes):
            torch.save(node.overlays[0].model_manager.model.state_dict(),
                       os.path.join(models_dir, "%d.model" % peer_ind))
        torch.save(avg_model.state_dict(), os.path.join(models_dir, "avg.model"))

    def checkpoint_model(self, peer_ind: int, round_nr: int):
        """
        Checkpoint a particular model of a peer during a particular round.
        """
        models_dir = os.path.join(self.data_dir, "models", "%d" % round_nr)
        os.makedirs(models_dir, exist_ok=True)

        model = self.nodes[peer_ind].overlays[0].model_manager.model
        torch.save(model.state_dict(), os.path.join(models_dir, "%d.model" % peer_ind))

    def test_models(self) -> Dict[int, Tuple[float, float]]:
        """
        Test the accuracy of all models in the model manager locally.
        """
        results: Dict[int, Tuple[float, float]] = {}
        for ind, model in enumerate(self.model_manager.incoming_trained_models.values()):
            self.logger.warning("Testing model %d on device %s..." % (ind + 1, self.args.accuracy_device_name))
            if not self.args.bypass_training:
                accuracy, loss = self.evaluator.evaluate_accuracy(model, device_name=self.args.accuracy_device_name)
            else:
                accuracy, loss = 0, 0

            results[ind] = (accuracy, loss)
        return results

    def get_statistics(self) -> Dict:
        # Determine both individual and aggregate statistics.
        total_bytes_up: int = 0
        total_bytes_down: int = 0
        total_train_time: float = 0
        total_network_time: float = 0

        individual_stats = {}
        for ind, node in enumerate(self.nodes):
            bytes_up = node.overlays[0].endpoint.bytes_up
            bytes_down = node.overlays[0].endpoint.bytes_down
            train_time = node.overlays[0].model_manager.model_trainer.total_training_time
            network_time = node.overlays[0].bw_scheduler.total_time_transmitting
            individual_stats[ind] = {
                "bytes_up": bytes_up,
                "bytes_down": bytes_down,
                "train_time": train_time,
                "network_time": network_time
            }

            total_bytes_up += bytes_up
            total_bytes_down += bytes_down
            total_train_time += train_time
            total_network_time += network_time

        aggregate_stats = {
            "bytes_up": total_bytes_up,
            "bytes_down": total_bytes_down,
            "train_time": total_train_time,
            "network_time": total_network_time
        }

        return {
            "time": asyncio.get_event_loop().time(),
            "global": aggregate_stats
        }

    def flush_statistics(self):
        """
        Flush all the statistics generated by nodes.
        """

        # Write away the model transfers between peers
        if self.args.bypass_model_transfers:
            with open(os.path.join(self.data_dir, "transfers.csv"), "a") as out_file:
                for node in self.nodes:
                    for transfer in node.overlays[0].transfers:
                        out_file.write("%s,%s,%d,%f,%f,%s,%d\n" % transfer)
                    node.overlays[0].transfers = []

        with open(os.path.join(self.data_dir, "statistics.json"), "a") as out_file:
            out_file.write(json.dumps(self.get_statistics()) + "\n")

    def on_simulation_finished(self) -> None:
        self.flush_statistics()

        if self.args.profile:
            yappi.stop()
            yappi_stats = yappi.get_func_stats()
            yappi_stats.sort("tsub")
            yappi_stats.save(os.path.join(self.data_dir, "yappi.stats"), type='callgrind')

    async def run(self) -> None:
        self.setup_directories()
        await self.start_ipv8_nodes()
        self.setup_logger()
        self.ipv8_discover_peers()
        self.apply_latencies()
        self.on_ipv8_ready()
        await self.setup_simulation()
        self.apply_traces()
        await self.start_simulation()
        self.on_simulation_finished()
