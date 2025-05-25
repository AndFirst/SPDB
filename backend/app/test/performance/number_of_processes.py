import asyncio
import json
from collections import defaultdict
from time import perf_counter

from app.path.service import PathService

from .utils import get_request_data, load_points


class TestNumProcesses:
    def __init__(self):
        self.number_of_processes_list = list(range(1, 17))
        self.points = load_points()
        self.request_data = get_request_data(load_points(), "Śródmieście", 10)

    async def run(self):
        times = defaultdict(list)
        for i in range(5):
            for number_of_processes in self.number_of_processes_list:
                print(f"Running test with {number_of_processes} processes")
                service = PathService(num_processes=number_of_processes)
                start_time = perf_counter()
                stats = await service.calculate_path(self.request_data)
                stats_dict = stats.model_dump()
                end_time = perf_counter()
                elapsed_time = end_time - start_time
                results = {
                    "elapsed_time": elapsed_time,
                    "optimized_stats": stats_dict["optimized_route"]["stats"],
                    "default_stats": stats_dict["default_route"]["stats"],
                }
                times[number_of_processes].append(results)
        with open("results/number_of_processes.json", "w") as f:
            json.dump(times, f, indent=4)


if __name__ == "__main__":
    test = TestNumProcesses()
    asyncio.run(test.run())
