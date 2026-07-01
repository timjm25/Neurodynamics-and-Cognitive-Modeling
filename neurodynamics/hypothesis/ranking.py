from typing import List

from .discovery import HypothesisCandidate


def rank_hypotheses(candidates: List[HypothesisCandidate],
                    novelty_weight: float = 0.5,
                    plausibility_weight: float = 0.5) -> List[HypothesisCandidate]:
    """Rank hypotheses by weighted score."""
    return sorted(candidates,
                  key=lambda h: h.score * (novelty_weight + plausibility_weight),
                  reverse=True)


def filter_by_confidence(candidates: List[HypothesisCandidate],
                          min_score: float = 0.5) -> List[HypothesisCandidate]:
    return [h for h in candidates if h.score >= min_score]
