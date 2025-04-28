# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/tree/main

import json
import logging
import argparse
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("openai").setLevel(logging.WARNING)

from lm_eval import tasks, evaluator, utils
from utils.setup import setup_model

def parse_args():
    parser = argparse.ArgumentParser()
    # LM Eval Harness 
    parser.add_argument(
        "--tasks", default=None)
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        help="Maximal batch size to try with --batch_size auto",
    )
    parser.add_argument("--output_path", default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)
    parser.add_argument("--sample_checkpoint", type=int, default=0)
    
    # Setup
    parser.add_argument('--model_name', type=str, default='llama2')
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--method", type=str, default='full')
    parser.add_argument('--target_sparsity', type=float, default=0.5)
    parser.add_argument('--prefill_ratio', type=float, default=0.1)
    parser.add_argument('--sparse_ratio', type=float, default=1)
    parser.add_argument("--config_file", type=str, default='config/llama-2-7b-hf_default.json')
    parser.add_argument("--sparse_config_file", type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_args()

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)
    
    if len(task_names) > 1:
        raise NotImplementedError

    _, tokenizer, model = setup_model(args)
    model = model.eval().to(args.device)

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    
    print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    results = evaluator.simple_evaluate(
        model=model,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        no_cache=args.no_cache,
        sample_checkpoint=args.sample_checkpoint,
        limit=args.limit,
        description_dict=description_dict,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        output_base_path=args.output_base_path,
        tokenizer=tokenizer,
    )
    
    print(evaluator.make_table(results))


if __name__ == "__main__":
    main()

