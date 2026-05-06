import copy
import re
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from trl import GRPOTrainer
from .Samplers import RandomSampler, SequentialSampler
from .utils import nanmax, nanmin
try:
    from vllm import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
except Exception:
    SamplingParams = None
    GuidedDecodingParams = None

def build_messages(problem: str, system_prompt: str, hint: str | None = None) -> list[dict[str, str]]:
    context_block = ""
    if hint:
        context_block = (
            "Context (exploration condition):\n"
            f"- Use this hint/approach: {hint}\n\n"
            "Important:\n"
            "- Follow this approach as your primary strategy.\n\n"
        )
    user_content = (
        "Problem:\n"
        f"{problem}\n\n"
        f"{context_block}"
        "Solve this step by step and provide your final numerical answer at the end."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def compute_advantage_with_groups(
    answer: Any,
    ys: list[str],
    cids: list[str],
    lbd: float,
    epsilon: float,
    reward_fn,
    device: torch.device,
):
    rewards = torch.tensor(
        [float(reward_fn(y, answer)) for y in ys],
        dtype=torch.float32,
        device=device,
    )
    N = rewards.numel()
    if N == 0:
        empty = torch.zeros(0, dtype=torch.float32, device=device)
        return empty, empty, None, None, None
    uniq = list(dict.fromkeys(cids))
    gid_map = {c: i for i, c in enumerate(uniq)}
    gids = torch.tensor([gid_map[c] for c in cids], device=device, dtype=torch.long)
    G = len(uniq)
    ones = torch.ones(N, device=device, dtype=torch.float32)
    group_sum = torch.zeros(G, device=device, dtype=torch.float32).index_add_(0, gids, rewards)
    group_cnt = torch.zeros(G, device=device, dtype=torch.float32).index_add_(0, gids, ones)
    group_mean = group_sum / group_cnt.clamp_min(1.0)

    overall_mean = rewards.mean()
    is_dropout = torch.tensor(
        [cid == "dropout" for cid in cids],
        dtype=torch.bool,
        device=device,
    )
    is_context = ~is_dropout

    advantages = torch.empty_like(rewards)
    if is_context.any():
        context_gids = gids[is_context]
        context_group_mean = group_mean[context_gids]
        context_rewards = rewards[is_context]
        advantages[is_context] = (
            (context_rewards - context_group_mean)
            + lbd * (context_group_mean - overall_mean)
        )
    if is_dropout.any():
        advantages[is_dropout] = rewards[is_dropout] - overall_mean
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + epsilon
    )

    return advantages, rewards, gids, group_mean, uniq
def compute_advantage_with_groups_dmean(
    answer: Any,
    ys: list[str],
    cids: list[str],
    lbd: float,
    epsilon: float,
    reward_fn,
    device: torch.device,
):
    rewards = torch.tensor(
        [float(reward_fn(y, answer)) for y in ys],
        dtype=torch.float32,
        device=device,
    )
    N = rewards.numel()
    if N == 0:
        empty = torch.zeros(0, dtype=torch.float32, device=device)
        return empty, empty, None, None, None
    uniq = list(dict.fromkeys(cids))
    gid_map = {c: i for i, c in enumerate(uniq)}
    gids = torch.tensor([gid_map[c] for c in cids], device=device, dtype=torch.long)
    G = len(uniq)
    ones = torch.ones(N, device=device, dtype=torch.float32)
    group_sum = torch.zeros(G, device=device, dtype=torch.float32).index_add_(0, gids, rewards)
    group_cnt = torch.zeros(G, device=device, dtype=torch.float32).index_add_(0, gids, ones)
    group_mean = group_sum / group_cnt.clamp_min(1.0)
    is_dropout = torch.tensor(
        [cid == "dropout" for cid in cids],
        dtype=torch.bool,
        device=device,
    )
    is_context = ~is_dropout
    if is_dropout.any():
        dropout_mean = rewards[is_dropout].mean()
    else:
        dropout_mean = rewards.mean()

    advantages = torch.empty_like(rewards)
    if is_context.any():
        context_gids = gids[is_context]
        context_group_mean = group_mean[context_gids]
        context_rewards = rewards[is_context]
        advantages[is_context] = (
            (context_rewards - context_group_mean)
            + lbd * (context_group_mean - dropout_mean)
        )
    if is_dropout.any():
        advantages[is_dropout] = rewards[is_dropout] - dropout_mean
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + epsilon
    )

    return advantages, rewards, gids, group_mean, uniq
def compute_advantage_with_dropout_baseline(
    answer: Any,
    ys: list[str],
    cids: list[str],
    epsilon: float,
    reward_fn,
    device: torch.device,
):
    rewards = torch.tensor(
        [float(reward_fn(y, answer)) for y in ys],
        dtype=torch.float32,
        device=device,
    )

    N = rewards.numel()
    if N == 0:
        empty = torch.zeros(0, dtype=torch.float32, device=device)
        return empty, empty, None, None, None
    uniq = list(dict.fromkeys(cids))
    gid_map = {c: i for i, c in enumerate(uniq)}
    gids = torch.tensor([gid_map[c] for c in cids], device=device, dtype=torch.long)
    G = len(uniq)
    ones = torch.ones(N, device=device, dtype=torch.float32)
    group_sum = torch.zeros(G, device=device, dtype=torch.float32).index_add_(0, gids, rewards)
    group_cnt = torch.zeros(G, device=device, dtype=torch.float32).index_add_(0, gids, ones)
    group_mean = group_sum / group_cnt.clamp_min(1.0)
    is_dropout = torch.tensor(
        [cid == "dropout" for cid in cids],
        dtype=torch.bool,
        device=device,
    )
    if is_dropout.any():
        dropout_mean = rewards[is_dropout].mean()
    else:
        dropout_mean = rewards.mean()

    advantages = rewards - dropout_mean
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + epsilon
    )

    return advantages, rewards, gids, group_mean, uniq
def compute_advantage_naive(
    answer: Any,
    ys: list[str],
    cids: list[str],
    lbd: float,
    epsilon: float,
    reward_fn,
    device: torch.device,
):
    rewards = torch.tensor(
        [float(reward_fn(y, answer)) for y in ys],
        dtype=torch.float32,
        device=device,
    )
    N = rewards.numel()
    if N == 0:
        empty = torch.zeros(0, dtype=torch.float32, device=device)
        return empty, empty, None, None, None

    uniq = list(dict.fromkeys(cids))
    gid_map = {c: i for i, c in enumerate(uniq)}
    gids = torch.tensor([gid_map[c] for c in cids], device=device, dtype=torch.long)
    G = len(uniq)

    ones = torch.ones(N, device=device, dtype=torch.float32)
    group_sum = torch.zeros(G, device=device, dtype=torch.float32).index_add_(0, gids, rewards)
    group_cnt = torch.zeros(G, device=device, dtype=torch.float32).index_add_(0, gids, ones)
    group_mean = group_sum / group_cnt.clamp_min(1.0)

    del lbd
    advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + epsilon)
    return advantages, rewards, gids, group_mean, uniq
class NudgeRLTrainer(GRPOTrainer):
    def __init__(
        self,
        *args,
        reward_fn=None,
        system_prompt: str = "You are a helpful assistant.",
        adv_lbd: float = 0.1,
        adv_eps: float = 1e-5,
        num_hint: int = 2,
        rollout_per_hint: int = 4,
        p_dropout: float = 0.25,
        distill_coeff: float = 0.01,
        sampler_type: str = "random",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert reward_fn is not None, "NudgeRLTrainer requires reward_fn=callable"
        self.reward_fn = reward_fn
        self.system_prompt = system_prompt


        self.adv_lbd = adv_lbd
        self.adv_eps = adv_eps
        self.num_hint = num_hint
        self.rollout_per_hint = rollout_per_hint
        self.context_key = "hints"
        self.p_dropout = p_dropout
        self.distill_coeff = distill_coeff
        

        if sampler_type == "random":
            self.sampler = RandomSampler(num_contexts=num_hint)
        elif sampler_type == "sequential":
            self.sampler = SequentialSampler(num_contexts=num_hint)
        else:
            raise ValueError(f"Unsupported sampler_type: {sampler_type}")

        target_ng = self.num_hint * self.rollout_per_hint
        if getattr(self, "num_generations", None) != target_ng:
            print(
                f"[NudgeRLTrainer] Auto-updating num_generations to {target_ng} (=M*K). "
                "Actual rollouts per example are variable due to dropout, but we won't rely on view(-1,num_generations)."
            )
            self.num_generations = target_ng
        custom_keys = [
            "context/intra_term_abs_mean",
            "context/inter_term_abs_mean",
            "context/dropout_reward_mean",
            "context/hinted_reward_mean",
            "context/hinted_minus_dropout",
            "context/context_reward_mean_gap",
            "adv_mean",
            "adv_std",
            "loss/grpo_x1",
            "loss/distill_x0",
            "loss/total",
        ]
        for mode in ("train", "eval"):
            for key in custom_keys:
                if key not in self._metrics[mode]:
                    self._metrics[mode][key] = []
        self._warned_missing_vllm_sync = False

    def _get_rollout_counts(self) -> List[int]:
        dropout_counts = np.random.binomial(self.rollout_per_hint, self.p_dropout, size=self.num_hint)
        rollout_counts = [int(dropout_counts.sum())]
        for c in dropout_counts:
            rollout_counts.append(int(self.rollout_per_hint - c))
        return rollout_counts
    def _vllm_generate_texts(
        self, prompts_text: List[str], n: int
    ) -> Tuple[List[List[int]], List[str], List[List[float]]]:
        assert self.use_vllm, "This helper is for vLLM only."
        device = self.accelerator.device
        if self.vllm_mode == "server":
            if self.accelerator.is_main_process:
                out = self.vllm_client.generate(
                    prompts=prompts_text,
                    images=None,
                    n=n,
                    repetition_penalty=self.repetition_penalty,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=-1 if self.top_k is None else self.top_k,
                    min_p=0.0 if self.min_p is None else self.min_p,
                    max_tokens=self.max_completion_length,
                    guided_decoding_regex=self.guided_decoding_regex,
                    generation_kwargs=self.args.generation_kwargs,
                )
                completion_ids = out["completion_ids"]
                completion_texts = out.get("completions", None)
                sampling_logprobs = out.get("logprobs", None)
            else:
                completion_ids, completion_texts, sampling_logprobs = None, None, None

            obj_list = [(completion_ids, completion_texts, sampling_logprobs)]
            torch.distributed.broadcast_object_list(obj_list, src=0)
            completion_ids, completion_texts, sampling_logprobs = obj_list[0]

            if completion_texts is None:
                completion_texts = self.processing_class.batch_decode(
                    [torch.tensor(ids, device=device) for ids in completion_ids],
                    skip_special_tokens=True,
                )
            if sampling_logprobs is None:
                sampling_logprobs = [[0.0] * len(ids) for ids in completion_ids]

            return completion_ids, completion_texts, sampling_logprobs
        assert SamplingParams is not None, "vllm SamplingParams not available"

        if self.guided_decoding_regex:
            guided_decoding = GuidedDecodingParams(regex=self.guided_decoding_regex)
        else:
            guided_decoding = None


        generation_kwargs = {
            "n": n,
            "repetition_penalty": self.repetition_penalty,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": -1 if self.top_k is None else self.top_k,
            "min_p": 0.0 if self.min_p is None else self.min_p,
            "max_tokens": self.max_completion_length,
            "guided_decoding": guided_decoding,
            "logprobs": 0,
        }
        if self.args.generation_kwargs is not None:
            generation_kwargs.update(self.args.generation_kwargs)

        sp = SamplingParams(**generation_kwargs)

        if getattr(self, "vllm_tensor_parallel_size", 1) != 1:
            raise RuntimeError(
                "NudgeRLTrainer variable-n in colocate currently assumes "
                "vllm_tensor_parallel_size == 1"
            )

        all_outputs = self.llm.generate(prompts_text, sampling_params=sp, use_tqdm=False)
        completion_ids = []
        completion_texts = []
        completion_logprobs = []
        for req in all_outputs:
            for o in req.outputs:
                completion_ids.append(o.token_ids)
                completion_texts.append(o.text)
                token_logprobs: List[float] = []
                if getattr(o, "logprobs", None):
                    for lp in o.logprobs:
                        if lp:
                            token_logprobs.append(float(next(iter(lp.values())).logprob))
                        else:
                            token_logprobs.append(0.0)
                if not token_logprobs:
                    token_logprobs = [0.0] * len(o.token_ids)
                completion_logprobs.append(token_logprobs)

        return completion_ids, completion_texts, completion_logprobs

    def _sync_vllm_weights_if_needed(self) -> None:
        if not self.use_vllm:
            return

        state = getattr(self, "state", None)
        global_step = int(getattr(state, "global_step", 0))
        last_loaded_step = int(getattr(self, "_last_loaded_step", -1))
        if global_step == last_loaded_step:
            return
        move_fn = getattr(self, "_move_model_to_vllm", None)
        if callable(move_fn):
            move_fn()
            self._last_loaded_step = global_step
            return
        vllm_generation = getattr(self, "vllm_generation", None)
        sync_fn = getattr(vllm_generation, "sync_weights", None) if vllm_generation is not None else None
        if callable(sync_fn):
            sync_fn()
            self._last_loaded_step = global_step
            return

        if not self._warned_missing_vllm_sync:
            print(
                "[NudgeRLTrainer] Warning: no known vLLM sync hook found; "
                "rollouts may use stale policy weights."
            )
            self._warned_missing_vllm_sync = True
    def _generate_and_score_completions(
        self, inputs: List[Dict[str, Union[torch.Tensor, Any]]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        from collections import defaultdict

        self._sync_vllm_weights_if_needed()

        dedup_factor = int(getattr(self, "num_generations", 1) or 1)
        if dedup_factor > 1 and len(inputs) >= dedup_factor and len(inputs) % dedup_factor == 0:
            generation_inputs = inputs[::dedup_factor]
        else:
            generation_inputs = inputs
        example_meta = []
        all_requests = []

        for ex_idx, x in enumerate(generation_inputs):
            answer = x.get("answer")
            if answer is None:
                raise KeyError("Each input must include 'answer' for reward_fn(y, answer).")

            problem = x.get("problem")
            if problem is None:
                p = x.get("prompt")
                if isinstance(p, str):
                    problem = p
                else:
                    raise KeyError("Provide either 'problem' (str) or 'prompt' (str).")

            sys_prompt = x.get("system_prompt", self.system_prompt)
            hints = x.get(self.context_key)
            hints = hints if isinstance(hints, list) else [hints] if hints is not None else []
            if hints is None:
                raise KeyError("Each input must include 'hints': list[str] length M.")
            if len(hints) > self.num_hint:
                hints = self.sampler.sample(hints)
            if len(hints) < self.num_hint:
                raise ValueError(f"Expected len(hints)==num_hint=={self.num_hint}, got {len(hints)}")
            counts = self._get_rollout_counts()
            group_hints = [None] + hints
            group_cids = ["dropout"] + [f"hint{i+1}" for i in range(self.num_hint)]
            base_msgs = build_messages(problem=problem, system_prompt=sys_prompt, hint=None)
            base_text = self.processing_class.apply_chat_template(
                base_msgs, tokenize=False, add_generation_prompt=True
            )
            t = self.processing_class(
                text=[base_text], return_tensors="pt",
                padding=False, add_special_tokens=False,
            )
            x0_ids = t["input_ids"][0].to(device)
            x0_mask = torch.ones_like(x0_ids, device=device)

            example_meta.append({"answer": answer, "x0_ids": x0_ids, "x0_mask": x0_mask})

            for hint_g, cid_g, n_g in zip(group_hints, group_cids, counts):
                if n_g <= 0:
                    continue
                if not self.use_vllm:
                    raise RuntimeError("This implementation currently expects use_vllm=True.")
                msgs = build_messages(problem=problem, system_prompt=sys_prompt, hint=hint_g)
                prompt_text = self.processing_class.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                ctx_t = self.processing_class(
                    text=[prompt_text], return_tensors="pt",
                    padding=False, add_special_tokens=False,
                )
                x1_ids = ctx_t["input_ids"][0].to(device)
                x1_mask = torch.ones_like(x1_ids, device=device)
                all_requests.append((ex_idx, cid_g, prompt_text, x1_ids, x1_mask, n_g))
        by_n: dict[int, list] = defaultdict(list)
        for req_idx, (_, _, prompt_text, _, _, n_g) in enumerate(all_requests):
            by_n[n_g].append((req_idx, prompt_text))

        req_results: dict[int, tuple] = {}
        for n_val, req_list in by_n.items():
            prompts = [pt for _, pt in req_list]
            ids_flat, texts_flat, logprobs_flat = self._vllm_generate_texts(prompts, n=n_val)
            for i, (req_idx, _) in enumerate(req_list):
                req_results[req_idx] = (
                    ids_flat[i * n_val : (i + 1) * n_val],
                    texts_flat[i * n_val : (i + 1) * n_val],
                    logprobs_flat[i * n_val : (i + 1) * n_val],
                )
        req_by_ex: dict[int, list] = defaultdict(list)
        for req_idx, (ex_idx, cid_g, _, x1_ids, x1_mask, n_g) in enumerate(all_requests):
            ids_list, texts_list, logprobs_list = req_results[req_idx]
            req_by_ex[ex_idx].append((cid_g, n_g, ids_list, texts_list, logprobs_list, x1_ids, x1_mask))
        x0_prompt_ids_all: List[torch.Tensor] = []
        x0_prompt_mask_all: List[torch.Tensor] = []
        x1_prompt_ids_all: List[torch.Tensor] = []
        x1_prompt_mask_all: List[torch.Tensor] = []
        completion_ids_all: List[List[int]] = []
        sampling_logprobs_all: List[List[float]] = []
        advantages_all: List[torch.Tensor] = []
        rewards_all: List[torch.Tensor] = []
        reward_group_mean_all: List[torch.Tensor] = []
        intra_term_abs_all: List[torch.Tensor] = []
        inter_term_abs_all: List[torch.Tensor] = []
        context_reward_mean_gap_all: List[torch.Tensor] = []
        dropout_reward_mean_all: List[torch.Tensor] = []
        hinted_reward_mean_all: List[torch.Tensor] = []
        hinted_minus_dropout_all: List[torch.Tensor] = []

        for ex_idx, meta in enumerate(example_meta):
            if ex_idx not in req_by_ex:
                continue

            ex_completion_ids: List[List[int]] = []
            ex_completion_texts: List[str] = []
            ex_sampling_logprobs: List[List[float]] = []
            ex_cids: List[str] = []
            ex_x1_ids: List[torch.Tensor] = []
            ex_x1_masks: List[torch.Tensor] = []

            for cid_g, n_g, ids_list, texts_list, logprobs_list, x1_ids, x1_mask in req_by_ex[ex_idx]:
                ex_completion_ids.extend(ids_list)
                ex_completion_texts.extend(texts_list)
                ex_sampling_logprobs.extend(logprobs_list)
                ex_cids.extend([cid_g] * n_g)
                ex_x1_ids.extend([x1_ids] * n_g)
                ex_x1_masks.extend([x1_mask] * n_g)

            if not ex_completion_texts:
                continue
            if len(ex_completion_ids) != len(ex_x1_ids):
                raise RuntimeError("Mismatched rollout prompt/completion count while building x1 training batch.")
            if len(ex_completion_ids) != len(ex_sampling_logprobs):
                raise RuntimeError("Mismatched completion and sampling logprob counts while building IS batch.")

            ex_advantages, ex_rewards, gids, group_mean, uniq = compute_advantage_with_groups(
                answer=meta["answer"],
                ys=ex_completion_texts,
                cids=ex_cids,
                lbd=self.adv_lbd,
                epsilon=self.adv_eps,
                reward_fn=self.reward_fn,
                device=device,
            )
            if gids is None:
                continue

            intra = ex_rewards - group_mean[gids]
            overall = ex_rewards.mean()
            inter = group_mean[gids] - overall
            gap = group_mean.max() - group_mean.min()

            dropout_gid = None
            for i, name in enumerate(uniq):
                if name == "dropout":
                    dropout_gid = i
                    break

            n_ex = len(ex_completion_texts)
            intra_term_abs_all.append(intra.abs())
            inter_term_abs_all.append(inter.abs())
            context_reward_mean_gap_all.append(gap.repeat(n_ex))
            reward_group_mean_all.append(ex_rewards.mean().repeat(n_ex))

            dropout_reward_mean = torch.tensor(0.0, dtype=torch.float32, device=device)
            hinted_reward_mean = torch.tensor(0.0, dtype=torch.float32, device=device)
            hinted_minus_dropout = torch.tensor(0.0, dtype=torch.float32, device=device)

            if dropout_gid is not None:
                is_dropout = (gids == dropout_gid)
                is_hinted = ~is_dropout
                if is_dropout.any():
                    dropout_reward_mean = ex_rewards[is_dropout].mean()
                if is_hinted.any():
                    hinted_reward_mean = ex_rewards[is_hinted].mean()
                if is_dropout.any() and is_hinted.any():
                    hinted_minus_dropout = hinted_reward_mean - dropout_reward_mean
            else:
                hinted_reward_mean = ex_rewards.mean()

            dropout_reward_mean_all.append(dropout_reward_mean.repeat(n_ex))
            hinted_reward_mean_all.append(hinted_reward_mean.repeat(n_ex))
            hinted_minus_dropout_all.append(hinted_minus_dropout.repeat(n_ex))

            x0_prompt_ids_all.extend([meta["x0_ids"]] * n_ex)
            x0_prompt_mask_all.extend([meta["x0_mask"]] * n_ex)
            x1_prompt_ids_all.extend(ex_x1_ids)
            x1_prompt_mask_all.extend(ex_x1_masks)
            completion_ids_all.extend(ex_completion_ids)
            sampling_logprobs_all.extend(ex_sampling_logprobs)
            advantages_all.append(ex_advantages)
            rewards_all.append(ex_rewards)

        if len(completion_ids_all) == 0:
            empty = torch.zeros((0, 0), dtype=torch.long, device=device)
            return {
                "prompt_ids": empty,
                "prompt_mask": empty,
                "base_prompt_ids": empty,
                "base_prompt_mask": empty,
                "completion_ids": empty,
                "completion_mask": empty,
                "advantages": torch.zeros((0,), dtype=torch.float32, device=device),
                "num_items_in_batch": torch.tensor(0, device=device),
            }

        def _left_pad(ids_list: List[torch.Tensor], mask_list: List[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            max_len = max(t.numel() for t in ids_list)
            padded_ids = torch.full(
                (len(ids_list), max_len), self.pad_token_id, dtype=torch.long, device=device
            )
            padded_mask = torch.zeros((len(mask_list), max_len), dtype=torch.long, device=device)
            for i, (ids, mask) in enumerate(zip(ids_list, mask_list)):
                L = ids.numel()
                padded_ids[i, max_len - L :] = ids
                padded_mask[i, max_len - L :] = mask
            return padded_ids, padded_mask

        prompt_ids, prompt_mask = _left_pad(x1_prompt_ids_all, x1_prompt_mask_all)
        base_prompt_ids, base_prompt_mask = _left_pad(x0_prompt_ids_all, x0_prompt_mask_all)

        completion_tensors = [torch.tensor(ids, dtype=torch.long, device=device) for ids in completion_ids_all]
        max_c = max(t.numel() for t in completion_tensors)
        completion_ids_tensor = torch.full(
            (len(completion_tensors), max_c), self.pad_token_id, dtype=torch.long, device=device
        )
        completion_mask = torch.zeros((len(completion_tensors), max_c), dtype=torch.long, device=device)
        for i, t in enumerate(completion_tensors):
            L = t.numel()
            completion_ids_tensor[i, :L] = t
            completion_mask[i, :L] = 1

        sampling_per_token_logps = torch.zeros(
            (len(completion_tensors), max_c), dtype=torch.float32, device=device
        )
        for i, token_logprobs in enumerate(sampling_logprobs_all):
            if not token_logprobs:
                continue
            L = min(len(token_logprobs), max_c)
            sampling_per_token_logps[i, :L] = torch.tensor(
                token_logprobs[:L], dtype=torch.float32, device=device
            )

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids_tensor], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        old_per_token_logps = None
        importance_sampling_ratio = None
        ref_per_token_logps = None
        with torch.no_grad():
            steps_per_generation = int(getattr(self.args, "steps_per_generation", 1) or 1)
            num_iterations = int(getattr(self, "num_iterations", 1) or 1)
            generate_every = max(1, steps_per_generation * num_iterations)
            need_old_logps = (self.args.gradient_accumulation_steps % generate_every != 0) or (
                self.use_vllm and self.vllm_importance_sampling_correction
            )
            if need_old_logps:
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    max_c,
                    compute_entropy=False,
                )
                if self.use_vllm and self.vllm_importance_sampling_correction:
                    importance_sampling_ratio = torch.exp(old_per_token_logps - sampling_per_token_logps)
                    importance_sampling_ratio = torch.clamp(
                        importance_sampling_ratio,
                        max=self.vllm_importance_sampling_cap,
                    )

            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        max_c,
                        compute_entropy=False,
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            max_c,
                            compute_entropy=False,
                        )

        advantages = torch.cat(advantages_all, dim=0)
        rewards = torch.cat(rewards_all, dim=0)
        num_rows = advantages.size(0)
        zeros_metric = torch.zeros((num_rows,), dtype=torch.float32, device=device)

        reward_group_mean_metric = (
            torch.cat(reward_group_mean_all, dim=0) if reward_group_mean_all else zeros_metric
        )
        intra_term_abs_metric = torch.cat(intra_term_abs_all, dim=0) if intra_term_abs_all else zeros_metric
        inter_term_abs_metric = torch.cat(inter_term_abs_all, dim=0) if inter_term_abs_all else zeros_metric
        context_reward_mean_gap_metric = (
            torch.cat(context_reward_mean_gap_all, dim=0) if context_reward_mean_gap_all else zeros_metric
        )
        dropout_reward_mean_metric = (
            torch.cat(dropout_reward_mean_all, dim=0) if dropout_reward_mean_all else zeros_metric
        )
        hinted_reward_mean_metric = (
            torch.cat(hinted_reward_mean_all, dim=0) if hinted_reward_mean_all else zeros_metric
        )
        hinted_minus_dropout_metric = (
            torch.cat(hinted_minus_dropout_all, dim=0) if hinted_minus_dropout_all else zeros_metric
        )

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "base_prompt_ids": base_prompt_ids,
            "base_prompt_mask": base_prompt_mask,
            "completion_ids": completion_ids_tensor,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": completion_mask.sum().to(device),
            "rewards": rewards.detach(),
            "reward_group_mean": reward_group_mean_metric.detach(),
            "context/intra_term_abs_mean": intra_term_abs_metric.detach(),
            "context/inter_term_abs_mean": inter_term_abs_metric.detach(),
            "context/dropout_reward_mean": dropout_reward_mean_metric.detach(),
            "context/hinted_reward_mean": hinted_reward_mean_metric.detach(),
            "context/hinted_minus_dropout": hinted_minus_dropout_metric.detach(),
            "context/context_reward_mean_gap": context_reward_mean_gap_metric.detach(),
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if importance_sampling_ratio is not None:
            output["importance_sampling_ratio"] = importance_sampling_ratio
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        return output
    def _compute_loss(self, model, inputs):
        mode = "train" if self.model.training else "eval"
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        base_prompt_ids = inputs.get("base_prompt_ids", prompt_ids)
        base_prompt_mask = inputs.get("base_prompt_mask", prompt_mask)

        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        advantages = inputs["advantages"]
        input_ids_x1 = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask_x1 = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps_x1, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids_x1,
            attention_mask_x1,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        if self.beta != 0.0:
            ref_per_token_logps = inputs.get("ref_per_token_logps")
            if ref_per_token_logps is None:
                if self.ref_model is not None:
                    with torch.no_grad():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.ref_model,
                            input_ids_x1,
                            attention_mask_x1,
                            logits_to_keep,
                            compute_entropy=False,
                            pixel_values=inputs.get("pixel_values"),
                            image_grid_thw=inputs.get("image_grid_thw"),
                            pixel_attention_mask=inputs.get("pixel_attention_mask"),
                            image_sizes=inputs.get("image_sizes"),
                        )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter(), torch.no_grad():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            input_ids_x1,
                            attention_mask_x1,
                            logits_to_keep,
                            compute_entropy=False,
                            pixel_values=inputs.get("pixel_values"),
                            image_grid_thw=inputs.get("image_grid_thw"),
                            pixel_attention_mask=inputs.get("pixel_attention_mask"),
                            image_sizes=inputs.get("image_sizes"),
                        )
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps_x1) - (ref_per_token_logps - per_token_logps_x1) - 1
            )

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps_x1.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps_x1 - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. "
                "Possible values are 'token' and 'sequence'."
            )

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if (
            self.use_vllm
            and self.vllm_importance_sampling_correction
            and "importance_sampling_ratio" in inputs
        ):
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            grpo_loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
            grpo_loss = grpo_loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            grpo_loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            grpo_loss = grpo_loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            grpo_loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            grpo_loss = grpo_loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dapo":
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            grpo_loss = (per_token_loss * completion_mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        input_ids_x0 = torch.cat([base_prompt_ids, completion_ids], dim=1)
        attention_mask_x0 = torch.cat([base_prompt_mask, completion_mask], dim=1)

        per_token_logps_x0, _ = self._get_per_token_logps_and_entropies(
            model,
            input_ids_x0,
            attention_mask_x0,
            logits_to_keep,
            compute_entropy=False,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
        )

        denom = completion_mask.sum(-1).clamp(min=1.0)
        seq_logp_x0 = (per_token_logps_x0 * completion_mask).sum(-1) / denom
        distill_loss = -(advantages * seq_logp_x0).mean()
        distill_loss = distill_loss / self.current_gradient_accumulation_steps

        loss = grpo_loss + self.distill_coeff * distill_loss
        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:
                return x.mean()
            return (x * completion_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())

        if "reward_group_mean" in inputs:
            rg = inputs["reward_group_mean"]
            gathered_rg = self.accelerator.gather(rg)
            self._metrics[mode]["reward"].append(gathered_rg.mean().item())
            self._metrics[mode]["reward_std"].append(gathered_rg.std(unbiased=False).item())
        elif "rewards" in inputs:
            r = inputs["rewards"]
            gathered_r = self.accelerator.gather(r)
            self._metrics[mode]["reward"].append(gathered_r.mean().item())
            self._metrics[mode]["reward_std"].append(gathered_r.std(unbiased=False).item())

        for context_key in [
            "context/intra_term_abs_mean",
            "context/inter_term_abs_mean",
            "context/dropout_reward_mean",
            "context/hinted_reward_mean",
            "context/hinted_minus_dropout",
            "context/context_reward_mean_gap",
        ]:
            if context_key in inputs:
                context_metric = inputs[context_key]
                self._metrics[mode][context_key].append(
                    self.accelerator.gather(context_metric.unsqueeze(0)).mean().item()
                )

        gathered_adv = self.accelerator.gather(advantages)
        self._metrics[mode]["adv_mean"].append(gathered_adv.mean().item())
        self._metrics[mode]["adv_std"].append(gathered_adv.std(unbiased=False).item())
        self._metrics[mode]["loss/grpo_x1"].append(self.accelerator.gather(grpo_loss.detach()).mean().item())
        self._metrics[mode]["loss/distill_x0"].append(self.accelerator.gather(distill_loss.detach()).mean().item())
        self._metrics[mode]["loss/total"].append(self.accelerator.gather(loss.detach()).mean().item())

        return loss
