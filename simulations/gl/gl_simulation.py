import os
import random
from argparse import Namespace
from asyncio import get_event_loop
from binascii import hexlify

from dlsim.core.model_manager import ModelManager
from dlsim.core.session_settings import LearningSettings, SessionSettings, GLSettings

from ipv8.configuration import ConfigBuilder
from ipv8.taskmanager import TaskManager

from simulations.learning_simulation import LearningSimulation


class GLSimulation(LearningSimulation):

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.task_manager = TaskManager()
        self.data_dir = os.path.join("data", "n_%d_%s_sd%d_gl" % (self.args.peers, self.args.dataset, self.args.seed))

    def get_ipv8_builder(self, peer_id: int) -> ConfigBuilder:
        builder = super().get_ipv8_builder(peer_id)
        if self.args.bypass_model_transfers:
            builder.add_overlay("GLBypassNetworkCommunity", "my peer", [], [], {}, [])
        else:
            builder.add_overlay("GLCommunity", "my peer", [], [], {}, [])
        return builder

    async def setup_simulation(self) -> None:
        await super().setup_simulation()

        # Setup the training process
        learning_settings = LearningSettings(
            learning_rate=self.args.learning_rate,
            momentum=self.args.momentum,
            batch_size=self.args.batch_size,
            weight_decay=self.args.weight_decay,
            local_steps=self.args.local_steps,
        )

        gl_settings = GLSettings(self.args.gl_round_timeout)

        self.session_settings = SessionSettings(
            work_dir=self.data_dir,
            dataset=self.args.dataset,
            learning=learning_settings,
            participants=[hexlify(node.overlays[0].my_peer.public_key.key_to_bin()).decode() for node in self.nodes],
            all_participants=[hexlify(node.overlays[0].my_peer.public_key.key_to_bin()).decode() for node in
                              self.nodes],
            target_participants=len(self.nodes),
            dataset_base_path=self.args.dataset_base_path,
            gl=gl_settings,
            model=self.args.model,
            alpha=self.args.alpha,
            partitioner=self.args.partitioner,
            eva_block_size=1000,
            is_simulation=True,
            train_device_name=self.args.train_device_name,
            bypass_training=self.args.bypass_training,
        )

        self.model_manager = ModelManager(None, self.session_settings, 0)

        for ind, node in enumerate(self.nodes):
            node.overlays[0].round_complete_callback = lambda round_nr, i=ind: self.on_round_complete(i, round_nr)
            node.overlays[0].setup(self.session_settings)

        self.build_topology()

        if self.args.accuracy_logging_interval > 0 and self.args.accuracy_logging_interval_is_in_sec:
            interval = self.args.accuracy_logging_interval
            self.logger.info("Registering logging interval task that triggers every %d seconds", interval)
            self.task_manager.register_task("acc_check", self.compute_all_accuracies, delay=interval, interval=interval)

    def compute_all_accuracies(self):
        cur_time = get_event_loop().time()

        tot_up, tot_down = 0, 0
        for node in self.nodes:
            tot_up += node.overlays[0].endpoint.bytes_up
            tot_down += node.overlays[0].endpoint.bytes_down

        self.logger.warning("Computing accuracies for all models, current time: %f, bytes up: %d, bytes down: %d",
                            cur_time, tot_up, tot_down)

        # Put all the models in the model manager
        eligible_nodes = []
        for ind, node in enumerate(self.nodes):
            if not self.nodes[ind].overlays[0].is_active:
                continue

            eligible_nodes.append((ind, node))

        # Don't test all models for efficiency reasons, just up to 20% of the entire network
        eligible_nodes = random.sample(eligible_nodes, min(len(eligible_nodes), int(len(self.nodes) * 0.2)))
        print("Will test accuracy of %d nodes..." % len(eligible_nodes))

        for ind, node in eligible_nodes:
            model = self.nodes[ind].overlays[0].model_manager.model
            self.model_manager.process_incoming_trained_model(b"%d" % ind, model)

        if self.args.dl_accuracy_method == "aggregate":
            if not self.args.bypass_training:
                avg_model = self.model_manager.aggregate_trained_models()
                accuracy, loss = self.evaluator.evaluate_accuracy(avg_model, device_name=self.args.accuracy_device_name)
            else:
                accuracy, loss = 0, 0

            with open(os.path.join(self.data_dir, "accuracies.csv"), "a") as out_file:
                out_file.write("%s,%d,%g,GL,%f,%d,%d,%f,%f\n" % (self.args.dataset, self.args.seed, self.args.learning_rate,
                                                                 get_event_loop().time(), 0, int(cur_time), accuracy, loss))
        elif self.args.dl_accuracy_method == "individual":
            # Compute the accuracies of all individual models
            results = self.test_models()

            for ind, acc_res in results.items():
                accuracy, loss = acc_res
                round_nr = self.nodes[ind].overlays[0].round
                with open(os.path.join(self.data_dir, "accuracies.csv"), "a") as out_file:
                    out_file.write("%s,%d,%g,GL,%f,%d,%d,%f,%f\n" %
                                   (self.args.dataset, self.args.seed, self.args.learning_rate,
                                    cur_time, ind, round_nr, accuracy, loss))

        self.model_manager.reset_incoming_trained_models()

    def build_topology(self):
        """
        Build a k-out topology. This is compatible with the experiment results in the GL papers.
        """
        for node in self.nodes:
            other_nodes = [n for n in self.nodes if n != node]
            node.overlays[0].nodes = other_nodes

    async def on_round_complete(self, peer_ind: int, round_nr: int):
        # Compute model accuracy
        if self.args.accuracy_logging_interval > 0 and not self.args.accuracy_logging_interval_is_in_sec and \
                round_nr % self.args.accuracy_logging_interval == 0:
            print("Will compute accuracy of peer %d for round %d!" % (peer_ind, round_nr))
            try:
                print("Testing model of peer %d on device %s..." % (peer_ind + 1, self.args.accuracy_device_name))
                model = self.nodes[peer_ind].overlays[0].model_manager.model
                if not self.args.bypass_training:
                    accuracy, loss = self.evaluator.evaluate_accuracy(
                        model, device_name=self.args.accuracy_device_name)
                else:
                    accuracy, loss = 0, 0

                with open(os.path.join(self.data_dir, "accuracies.csv"), "a") as out_file:
                    out_file.write("%s,%d,%g,GL,%f,%d,%d,%f,%f\n" %
                                   (self.args.dataset, self.args.seed, self.args.learning_rate,
                                    get_event_loop().time(), peer_ind, round_nr, accuracy, loss))
            except ValueError as e:
                print("Encountered error during evaluation check - dumping all models and stopping")
                self.checkpoint_models(round_nr)
                raise e

        # Checkpoint the model
        if self.args.checkpoint_interval and round_nr % self.args.checkpoint_interval == 0:
            self.checkpoint_model(peer_ind, round_nr)
