import asyncio
from time import perf_counter

async def print_even_numbers():
    for n in range(0, 10, 2):
        print(n)
        await asyncio.sleep(1)

async def print_odd_numbers():
    for n in range(1, 10, 2):
        print(n)
        await asyncio.sleep(1)

async def main():
    start_time = perf_counter()
    results = asyncio.as_completed([print_even_numbers(), print_odd_numbers()])
    for result in results:
        await result
    end_time = perf_counter()
    print(f"use_time:{end_time - start_time : .2f}")

asyncio.run(main())