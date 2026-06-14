import verifiers as vf
from verifiers.utils.data_utils import (
    BOXED_SYSTEM_PROMPT,
    extract_boxed_answer,
    load_example_dataset,
)


def load_environment(**kwargs) -> vf.Environment:
    """
    Loads a custom environment.
    """
    parser = vf.Parser(extract_fn=extract_boxed_answer)

    def answer_reward_func(parser, completion, answer, **kwargs):
        response = parser.parse_answer(completion) or ""
        return 1.0 if response == answer else 0.0

    rubric = vf.Rubric(
        parser=parser,
        funcs=[answer_reward_func, parser.get_format_reward_func()],
        weights=[1.0, 0.0],
    )
    train_dataset = load_example_dataset("aime2025")
    env = vf.SingleTurnEnv(
        dataset=train_dataset,
        system_prompt=BOXED_SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
    )
    return env
