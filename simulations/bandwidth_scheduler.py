"""
These modules are responsible for scheduling model transfers while adhering to bandwidth limitations of both the
sending and receiving party.
"""
import logging
import random
from asyncio import Future, get_event_loop, InvalidStateError
from typing import List, Dict

from ipv8.taskmanager import TaskManager


class BWScheduler(TaskManager):

    def __init__(self, peer_pk: bytes, my_id: str) -> None:
        super().__init__()
        self.bw_limit: int = 0  # in bytes/s
        self.outgoing_requests: List[Transfer] = []  # Outgoing transfers waiting to be started
        self.incoming_requests: List[Transfer] = []  # Incoming transfers waiting to be started

        self.outgoing_transfers: List[Transfer] = []  # Ongoing outgoing transfers
        self.incoming_transfers: List[Transfer] = []  # Ongoing incoming transfers

        self.peer_pk = peer_pk
        self.my_id = my_id

        self.logger = logging.getLogger(self.__class__.__name__)

        self.is_active: bool = False  # Whether we are sending or receiving something
        self.became_active: float = 0
        self.total_time_transmitting: float = 0

    def get_allocated_outgoing_bw(self) -> int:
        allocated_bw: int = sum([transfer.allocated_bw for transfer in self.outgoing_transfers])
        assert allocated_bw <= self.bw_limit, "Allocated outgoing bandwidth of %s (%d) cannot exceed limit (%d)" % (self.my_id, allocated_bw, self.bw_limit)
        return allocated_bw

    def get_allocated_incoming_bw(self) -> int:
        allocated_bw: int = sum([transfer.allocated_bw for transfer in self.incoming_transfers])
        assert allocated_bw <= self.bw_limit, "Allocated incoming bandwidth of %s (%d) cannot exceed limit (%d)" % (self.my_id, allocated_bw, self.bw_limit)
        return allocated_bw

    def register_transfer(self, transfer, is_outgoing=False):
        if not self.incoming_transfers and not self.outgoing_transfers:
            self.is_active = True
            self.became_active = get_event_loop().time()

        if is_outgoing:
            self.outgoing_transfers.append(transfer)
        else:
            self.incoming_transfers.append(transfer)

    def unregister_transfer(self, transfer, is_outgoing=False):
        if is_outgoing:
            if transfer in self.outgoing_transfers:
                self.outgoing_transfers.remove(transfer)
        else:
            if transfer in self.incoming_transfers:
                self.incoming_transfers.remove(transfer)

        if not self.incoming_transfers and not self.outgoing_transfers:
            self.is_active = False
            self.total_time_transmitting += (get_event_loop().time() - self.became_active)

    def add_transfer(self, receiver_scheduler: "BWScheduler", transfer_size: int) -> "Transfer":
        """
        A new transfer request arrived.
        :param transfer_size: Size of the transfer, in bytes
        """
        transfer: Transfer = Transfer(self, receiver_scheduler, transfer_size)
        self.outgoing_requests.append(transfer)
        self.logger.debug("Adding transfer request %d: %s => %s to the queue", transfer.transfer_id, self.my_id,
                          transfer.receiver_scheduler.my_id)
        self.schedule()

        return transfer

    def schedule(self):
        """
        Try to schedule pending outgoing requests and allocate bandwidth to them.
        """
        sender_bw_left: int = self.bw_limit - self.get_allocated_outgoing_bw()
        if sender_bw_left == 0:
            return  # Cannot accept more pending requests

        requests_scheduled: List[Transfer] = []
        for request in self.outgoing_requests:
            receiver_bw_left = request.receiver_scheduler.bw_limit - request.receiver_scheduler.get_allocated_incoming_bw()
            bw_to_allocate = min(sender_bw_left, receiver_bw_left)
            if bw_to_allocate > 0:
                self.schedule_request(request, bw_to_allocate)
                requests_scheduled.append(request)
                sender_bw_left = self.bw_limit - self.get_allocated_outgoing_bw()  # Update this as it has changed

                # Do we have outgoing bandwidth left to allocate more requests?
                if sender_bw_left == 0:
                    break  # Cannot accept more pending requests
            else:
                # Add this transfer as pending request in the queue of the receiver, try again later.
                if request not in request.receiver_scheduler.incoming_requests:
                    self.logger.debug("Sender %s adding transfer %d as pending incoming request in the scheduler of "
                                      "receiver %s", self.my_id, request.transfer_id, request.receiver_scheduler.my_id)
                    request.receiver_scheduler.incoming_requests.append(request)

        for request in requests_scheduled:
            self.outgoing_requests.remove(request)

    def schedule_request(self, request, bw_to_allocate: int):
        """
        Schedule a particular request - we know for sure that there is bandwidth available for this transfer.
        """
        self.logger.debug("Starting transfer %d: %s => %s (allocated %d bw to this transfer, s %d/%d, r %d/%d)", request.transfer_id, self.my_id,
                          request.receiver_scheduler.my_id, bw_to_allocate, self.get_allocated_outgoing_bw(), self.bw_limit, request.receiver_scheduler.get_allocated_incoming_bw(), request.receiver_scheduler.bw_limit)
        request.allocated_bw = bw_to_allocate
        estimated_finish_time = request.transfer_size / request.allocated_bw
        request.start_time = get_event_loop().time()
        request.last_time_updated = get_event_loop().time()
        task_name = "transfer_%d_finish_%d" % (request.transfer_id, request.reschedules)
        self.register_task(task_name, self.on_outgoing_transfer_complete, request, delay=estimated_finish_time)
        self.register_transfer(request, is_outgoing=True)
        request.receiver_scheduler.register_transfer(request, is_outgoing=False)
        if request in request.receiver_scheduler.incoming_requests:
            request.receiver_scheduler.incoming_requests.remove(request)

    def on_outgoing_transfer_complete(self, transfer):
        """
        An outgoing transfer has completed.
        """
        self.logger.debug("Transfer %d: %s => %s has completed", transfer.transfer_id, self.my_id,
                          transfer.receiver_scheduler.my_id)
        transfer.finish()

        # Inform the other side
        self.unregister_transfer(transfer, is_outgoing=True)
        transfer.receiver_scheduler.on_incoming_transfer_complete(transfer)

        # Try to schedule remaining requests as we might have unallocated bandwidth at this point.
        self.schedule()

    def on_outgoing_transfer_failed(self, failed_transfer):
        self.unregister_transfer(failed_transfer, is_outgoing=True)
        self.cancel_pending_task("transfer_%d_finish_%d" % (failed_transfer.transfer_id, failed_transfer.reschedules))
        self.schedule()

    def on_incoming_transfer_complete(self, completed_transfer):
        """
        An incoming transfer has been completed.
        We first try to allocate more bandwidth to our ongoing requests.
        Then we inform other pending incoming requests.
        """
        self.unregister_transfer(completed_transfer, is_outgoing=False)

        # Prioritize allocating bandwidth to ongoing transfers
        for transfer in self.incoming_transfers + self.incoming_requests:
            self.logger.debug("Informing sender %s about available bandwidth for transfer %d",
                              transfer.sender_scheduler.my_id, transfer.transfer_id)
            transfer.sender_scheduler.on_receiver_inform_about_free_bandwidth(transfer)

            incoming_bw_left: int = self.bw_limit - self.get_allocated_incoming_bw()
            if incoming_bw_left == 0:
                break

    def on_receiver_inform_about_free_bandwidth(self, transfer):
        """
        A receiver of a pending transfer has informed us (the sender) about newly available bandwidth for a particular
        transfer. Adjust this transfer and try to allocate more if we can.
        """
        sender_bw_left: int = self.bw_limit - self.get_allocated_outgoing_bw()
        receiver_bw_left: int = transfer.receiver_scheduler.bw_limit - transfer.receiver_scheduler.get_allocated_incoming_bw()

        # This is either an ongoing request or a pending request
        if transfer in self.outgoing_transfers:
            self.logger.debug("Sender %s got available bw notification from receiver %s for ongoing transfer %s",
                              self.my_id, transfer.receiver_scheduler.my_id, transfer.transfer_id)
            # It's an ongoing transfer, increase the allocated bw of this transfer accordingly
            additional_bw_to_allocate = min(sender_bw_left, receiver_bw_left)
            if additional_bw_to_allocate > 0:
                # We can allocate more bw to this transfer, do so and update everything accordingly.
                self.logger.debug("Allocating %d additional bw to transfer %d", additional_bw_to_allocate,
                                  transfer.transfer_id)
                task_name = "transfer_%d_finish_%d" % (transfer.transfer_id, transfer.reschedules)
                self.cancel_pending_task(task_name)

                # First we update how much of the transfer has been completed at this point.
                transfer.update()

                # "Restart" the transfer and reschedule the completion event
                transfer.allocated_bw += additional_bw_to_allocate
                new_estimated_finish_time = (transfer.transfer_size - transfer.transferred) / transfer.allocated_bw
                transfer.reschedules += 1
                new_task_name = "transfer_%d_finish_%d" % (transfer.transfer_id, transfer.reschedules)
                self.register_task(new_task_name, self.on_outgoing_transfer_complete, transfer,
                                   delay=new_estimated_finish_time)
        elif transfer in self.outgoing_requests:
            self.logger.debug("Sender %s got available bw notification from receiver %s for pending request %s",
                              self.my_id, transfer.receiver_scheduler.my_id, transfer.transfer_id)
            bw_to_allocate = min(sender_bw_left, receiver_bw_left)
            if bw_to_allocate > 0:
                self.schedule_request(transfer, bw_to_allocate)
                self.outgoing_requests.remove(transfer)
        else:
            raise RuntimeError("We do not know about request %d!" % transfer.transfer_id)

    def kill_all_transfers(self):
        transfer_count: int = len(self.incoming_transfers) + len(self.outgoing_transfers)
        if transfer_count > 0:
            self.logger.warning("Interrupting all %d transfers of participant %s in the scheduler",
                                transfer_count, self.my_id)
        self.cancel_all_pending_tasks()
        for transfer in self.outgoing_transfers:
            transfer.receiver_scheduler.on_incoming_transfer_complete(transfer)
            self.logger.debug("Failing outgoing transfer %d: %s => %s", transfer.transfer_id, self.my_id,
                              transfer.receiver_scheduler.my_id)
            transfer.fail()
        for transfer in self.incoming_transfers:
            transfer.sender_scheduler.on_outgoing_transfer_failed(transfer)
            self.logger.debug("Failing incoming transfer %d: %s => %s", transfer.transfer_id, self.my_id,
                              transfer.receiver_scheduler.my_id)
            transfer.fail()

        # Clean up all the pending requests
        for request in self.outgoing_requests:
            if request in request.receiver_scheduler.incoming_requests:
                request.receiver_scheduler.incoming_requests.remove(request)
        for request in self.incoming_requests:
            if request in request.sender_scheduler.outgoing_requests:
                request.sender_scheduler.outgoing_requests.remove(request)

        self.incoming_transfers = []
        self.outgoing_transfers = []
        self.incoming_requests = []
        self.outgoing_requests = []


class Transfer:
    """
    Represents a bandwidth transfer.
    """

    def __init__(self, sender_scheduler: BWScheduler, receiver_scheduler: BWScheduler, transfer_size: int):
        self.transfer_id = random.randint(0, 100000000000)
        self.sender_scheduler: BWScheduler = sender_scheduler
        self.receiver_scheduler: BWScheduler = receiver_scheduler
        self.transfer_size: int = transfer_size
        self.transferred: int = 0
        self.allocated_bw: int = 0
        self.start_time: int = -1
        self.last_time_updated: int = 0
        self.complete_future: Future = Future()
        self.reschedules: int = 0
        self.metadata: Dict = {}

    def finish(self):
        self.update()
        try:
            self.complete_future.set_result(None)
        except InvalidStateError:
            self.sender_scheduler.logger.error("Finish of transfer %s resulted in an InvalidStateError - "
                                               "ignoring for now", self)

    def fail(self):
        self.update()
        try:
            self.complete_future.set_exception(RuntimeError("Transfer interrupted"))
        except InvalidStateError:
            self.sender_scheduler.logger.error("Failure of transfer %s resulted in an InvalidStateError - "
                                               "ignoring for now", self)

    def update(self):
        transferred: float = (get_event_loop().time() - self.last_time_updated) * self.allocated_bw
        self.transferred += transferred
        self.last_time_updated = get_event_loop().time()

    def get_transferred_bytes(self) -> int:
        return self.transferred

    def __str__(self):
        return "%s (%s => %s)" % (self.transfer_id, self.sender_scheduler.my_id, self.receiver_scheduler.my_id)
