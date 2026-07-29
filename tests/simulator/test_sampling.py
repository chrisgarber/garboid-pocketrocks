from __future__ import annotations

import pytest

from garboid_pocketrocks.rules import (
    LIVE_RULESET,
    VALUE_CHARTS,
    PlayerSetup,
    RulesetValidationError,
    live_ruleset,
)
from garboid_pocketrocks.simulator.sampling import (
    FixedRulesetSampler,
    RulesetSampler,
    RulesetVariationSampler,
    WeightedRulesetSampler,
    derive_seed,
)


def test_fixed_sampler_always_returns_its_single_supported_ruleset() -> None:
    sampler = FixedRulesetSampler(LIVE_RULESET)

    assert isinstance(sampler, RulesetSampler)
    assert sampler.support() == (LIVE_RULESET,)
    assert sampler.sample(root_seed=1, game_index=99) is LIVE_RULESET


def test_weighted_sampler_is_seeded_by_game_index() -> None:
    sampler = WeightedRulesetSampler(
        ((live_ruleset("A"), 1), (live_ruleset("E"), 2))
    )

    assert [sampler.sample(root_seed=7, game_index=i) for i in range(20)] == [
        sampler.sample(root_seed=7, game_index=i) for i in range(20)
    ]


def test_weighted_sampler_support_is_distinct_and_ordered() -> None:
    chart_a = live_ruleset("A")
    chart_e = live_ruleset("E")
    sampler = WeightedRulesetSampler(
        ((chart_a, 1), (chart_e, 2), (chart_a, 3))
    )

    assert sampler.support() == (chart_a, chart_e)


@pytest.mark.parametrize("weight", [0, -1, 1.5, True])
def test_weighted_sampler_rejects_non_positive_integer_weights(
    weight: object,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        WeightedRulesetSampler(((LIVE_RULESET, weight),))  # type: ignore[arg-type]


def test_variation_sampler_can_change_each_public_axis() -> None:
    sampler = RulesetVariationSampler(
        base=LIVE_RULESET,
        resource_count_options=((6, 6, 6, 6, 6), (5, 6, 7, 6, 6)),
        action_count_options=(
            LIVE_RULESET.action_counts,
            (14, 7, 3, 2, 2, 2),
        ),
        setup_options=(
            LIVE_RULESET.player_setups,
            (
                PlayerSetup(3, 35, 4),
                PlayerSetup(4, 28, 3),
                PlayerSetup(5, 22, 2),
            ),
        ),
        value_chart_options=(VALUE_CHARTS["A"], VALUE_CHARTS["E"]),
        objective_options=(
            (LIVE_RULESET.objective_pool, 4, True),
            (LIVE_RULESET.objective_pool[:12], 3, True),
        ),
    )

    samples = {sampler.sample(root_seed=13, game_index=i) for i in range(100)}

    assert len({sample.resource_counts for sample in samples}) == 2
    assert len({sample.action_counts for sample in samples}) == 2
    assert len({sample.player_setups for sample in samples}) == 2
    assert len({sample.value_chart for sample in samples}) == 2
    assert len(
        {
            (
                sample.objective_pool,
                sample.active_objective_count,
                sample.objectives_enabled,
            )
            for sample in samples
        }
    ) == 2


def test_variation_support_is_the_validated_cartesian_product() -> None:
    sampler = RulesetVariationSampler(
        base=LIVE_RULESET,
        resource_count_options=(LIVE_RULESET.resource_counts,),
        action_count_options=(LIVE_RULESET.action_counts,),
        setup_options=(LIVE_RULESET.player_setups,),
        value_chart_options=(VALUE_CHARTS["A"], VALUE_CHARTS["E"]),
        objective_options=(
            (LIVE_RULESET.objective_pool, 4, True),
            ((), 0, False),
        ),
    )

    assert len(sampler.support()) == 4
    assert len({ruleset.name for ruleset in sampler.support()}) == 4


def test_variation_names_are_stable_and_content_addressed() -> None:
    options = {
        "base": LIVE_RULESET,
        "resource_count_options": (LIVE_RULESET.resource_counts,),
        "action_count_options": (LIVE_RULESET.action_counts,),
        "setup_options": (LIVE_RULESET.player_setups,),
        "value_chart_options": (VALUE_CHARTS["A"], VALUE_CHARTS["E"]),
        "objective_options": ((LIVE_RULESET.objective_pool, 4, True),),
    }

    first = RulesetVariationSampler(**options)  # type: ignore[arg-type]
    second = RulesetVariationSampler(**options)  # type: ignore[arg-type]

    assert [ruleset.name for ruleset in first.support()] == [
        ruleset.name for ruleset in second.support()
    ]
    assert all(
        ruleset.name.startswith("live-A-") and len(ruleset.name.rsplit("-", 1)[1]) == 12
        for ruleset in first.support()
    )


def test_variation_sampler_rejects_empty_options() -> None:
    with pytest.raises(ValueError, match="resource_count_options"):
        RulesetVariationSampler(
            base=LIVE_RULESET,
            resource_count_options=(),
            action_count_options=(LIVE_RULESET.action_counts,),
            setup_options=(LIVE_RULESET.player_setups,),
            value_chart_options=(LIVE_RULESET.value_chart,),
            objective_options=((LIVE_RULESET.objective_pool, 4, True),),
        )


def test_variation_sampler_validates_every_supported_combination() -> None:
    invalid_actions = (1, 0, 0, 0, 0, 0)

    with pytest.raises(RulesetValidationError, match="auction capacity"):
        RulesetVariationSampler(
            base=LIVE_RULESET,
            resource_count_options=(LIVE_RULESET.resource_counts,),
            action_count_options=(LIVE_RULESET.action_counts, invalid_actions),
            setup_options=(LIVE_RULESET.player_setups,),
            value_chart_options=(LIVE_RULESET.value_chart,),
            objective_options=((LIVE_RULESET.objective_pool, 4, True),),
        )


def test_seed_derivation_is_stable_and_namespaced() -> None:
    assert derive_seed(7, "ruleset", 3) == derive_seed(7, "ruleset", 3)
    assert derive_seed(7, "ruleset", 3) != derive_seed(7, "game", 3)
