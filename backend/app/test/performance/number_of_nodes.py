import asyncio
import json
from collections import defaultdict
from time import perf_counter

from app.path.service import PathService

from .utils import get_request_data, load_points


class TestNumNodes:
    def __init__(self):
        self.points = load_points()

    async def run(self):
        times = defaultdict(list)
        for i in range(5):
            for num_nodes in range(2, 11):
                print(f"Running test with {num_nodes} nodes")
                request = get_request_data(self.points, "Śródmieście", num_nodes)
                service = PathService(num_processes=4)
                start_time = perf_counter()
                stats = await service.calculate_path(request)
                stats_dict = stats.model_dump()
                end_time = perf_counter()
                elapsed_time = end_time - start_time
                results = {
                    "elapsed_time": elapsed_time,
                    "optimized_stats": stats_dict["optimized_route"]["stats"],
                    "default_stats": stats_dict["default_route"]["stats"],
                }
                times[num_nodes].append(results)
        with open("results/number_of_nodes.json", "w") as f:
            json.dump(times, f, indent=4)


if __name__ == "__main__":
    test = TestNumNodes()
    asyncio.run(test.run())
