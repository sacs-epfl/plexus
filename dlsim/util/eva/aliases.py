from typing import Callable, Coroutine

from ipv8.types import Peer

from dlsim.util.eva.exceptions import TransferException
from dlsim.util.eva.result import TransferResult

TransferCompleteCallback = Callable[[TransferResult], Coroutine]
TransferErrorCallback = Callable[[Peer, TransferException], Coroutine]
TransferRequestCallback = Callable[[Peer, bytes], Coroutine]
