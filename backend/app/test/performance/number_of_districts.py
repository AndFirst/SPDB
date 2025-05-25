import asyncio
import json
import os
from collections import defaultdict
from time import perf_counter

from app.path.service import PathService

from .utils import get_districts, get_one_point_per_district, load_points


class TestDistance:
    def __init__(self):
        self.points = load_points()
        self.districts = get_districts(self.points)

    async def run(self):
        times = defaultdict(list)
        for i in range(5):
            for num_nodes in range(2, len(self.districts) + 1):
                print(f"Running test with {num_nodes} districts")
                request = get_one_point_per_district(
                    self.points, self.districts[:num_nodes]
                )
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
        with open("results/number_of_districts.json", "w") as f:
            json.dump(times, f, indent=4)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    print("Starting TestDistance")
    test_distance = TestDistance()
    asyncio.run(test_distance.run())
    print("Finished TestDistance")
