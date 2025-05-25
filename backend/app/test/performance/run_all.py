import asyncio
import os

from .distances import TestDistance
from .number_of_nodes import TestNumNodes
from .number_of_processes import TestNumProcesses

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    print("Starting TestDistance")
    test_distance = TestDistance()
    asyncio.run(test_distance.run())
    print("Finished TestDistance")

    print("Starting TestNumNodes")
    test_num_nodes = TestNumNodes()
    asyncio.run(test_num_nodes.run())
    print("Finished TestNumNodes")

    print("Starting TestNumProcesses")
    test_num_processes = TestNumProcesses()
    asyncio.run(test_num_processes.run())
    print("Finished TestNumProcesses")
