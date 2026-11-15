import asyncio

from experiments.experiment_aligner import ExperimentAligner
from experiments.experiment_runner import ExperimentRunner
from experiments.generic_summaries.context import GenericSummaryContext
from experiments.query_based_summaries.context import QueryBasedSummaryContext


async def main():
    contexts = [
        GenericSummaryContext(),
        QueryBasedSummaryContext(),
    ]
    for context in contexts:
        await ExperimentRunner(context, n_parallel=2).run(sequential=True)
        await ExperimentAligner(context, enable_embedding=True, enable_llm=False).align()
        await ExperimentAligner(context, enable_embedding=True, enable_llm=True).align()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
