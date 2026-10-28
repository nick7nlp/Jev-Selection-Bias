"""Score newly collected selections without changing the question set."""

from .protocol import ORDER_ARMS, REPEAT_ARMS


def score(questions: list[dict], predictions: list[dict], arms: tuple[str, ...], orders: dict) -> dict:
    """Score ordered requests after mapping all selections to candidate IDs."""
    by_id = {q["id"]: q for q in questions}
    indexed = {}
    models = set()
    for row in predictions:
        model, qid, arm = row["model"], row["question_id"], row["arm"]
        if qid not in by_id or arm not in orders[qid]:
            raise ValueError("prediction refers to an unknown question or arm")
        key = (model, qid, arm)
        if key in indexed:
            raise ValueError("duplicate prediction")
        if row.get("order") != orders[qid][arm]:
            raise ValueError("prediction order does not match the manifest")
        selected = row.get("candidate_id")
        if selected is not None and selected not in by_id[qid]["candidate_ids"]:
            raise ValueError("prediction has an unknown candidate ID")
        if selected is not None and row.get("status") != "ok":
            raise ValueError("non-ok prediction contains a selection")
        indexed[key] = selected
        models.add(model)
    if not models:
        raise ValueError("no predictions to score")
    n = len(questions)
    output = {"question_count": n, "models": {}}
    for model in sorted(models):
        selected = {arm: [indexed.get((model, q["id"], arm)) for q in questions] for arm in arms}
        gold = [q["candidate_ids"][q["gold_index"]] for q in questions]
        correct = {arm: [a is not None and a == g for a, g in zip(selected[arm], gold)] for arm in arms}
        stats = {"accuracy": {arm: sum(correct[arm]) / n for arm in arms}}
        families = {"reordered": ORDER_ARMS, "original_repeats": REPEAT_ARMS}
        for family, family_arms in families.items():
            if not set(family_arms) <= set(arms):
                continue
            failed, wrong_questions, wrong_pairs = set(), set(), set()
            curve = []
            for step, arm in enumerate(family_arms):
                for i, q in enumerate(questions):
                    choice = selected[arm][i]
                    if not correct[arm][i]:
                        failed.add(q["id"])
                    if choice is not None and choice != gold[i]:
                        wrong_questions.add(q["id"])
                        wrong_pairs.add((q["id"], choice))
                prefix = family_arms[:step + 1]
                complete = [i for i in range(n) if all(selected[a][i] is not None for a in prefix)]
                agree = sum(len({selected[a][i] for a in prefix}) == 1 for i in complete)
                initial_correct = [i for i in range(n) if correct["original"][i]]
                retained = sum(all(correct[a][i] for a in prefix) for i in initial_correct)
                curve.append({
                    "arm": arm,
                    "cumulative_failed_questions": len(failed),
                    "cumulative_questions_with_wrong_selection": len(wrong_questions),
                    "cumulative_distinct_wrong_candidates": len(wrong_pairs),
                    "additional_wrong_candidates": len(wrong_pairs) - len(wrong_questions),
                    "complete_selection_questions": len(complete),
                    "cumulative_candidate_agreement": agree / n,
                    "initially_correct_questions": len(initial_correct),
                    "retained_initial_accuracy": retained / len(initial_correct) if initial_correct else None,
                })
            stats[family] = curve
        output["models"][model] = stats
    return output
