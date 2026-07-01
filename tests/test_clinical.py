import pytest

from neurodynamics.agents import BrainNetwork
from neurodynamics.clinical import (
    AlzheimersDisease, ParkinsonsDisease, Schizophrenia, Epilepsy, ADHD,
)


def _make_network():
    return BrainNetwork()


def test_alzheimers_reduces_hippocampal_connectivity():
    net = _make_network()
    hippo = net.regions["Hippocampus"]
    initial_scale = hippo.connectivity_scale
    ad = AlzheimersDisease(stage=0.7)
    ad.apply(net)
    assert hippo.connectivity_scale < initial_scale


def test_parkinsons_alters_basal_ganglia():
    net = _make_network()
    bg = net.regions["BasalGanglia"]
    initial_dopamine = bg.dopamine
    pd = ParkinsonsDisease(severity=0.6)
    pd.apply(net)
    assert bg.dopamine < initial_dopamine


def test_schizophrenia_changes_pfc_connectivity():
    net = _make_network()
    pfc = net.regions["PFC"]
    initial_gain = pfc.control_gain
    sz = Schizophrenia(severity=0.5)
    sz.apply(net)
    assert pfc.control_gain < initial_gain


def test_epilepsy_lowers_seizure_threshold():
    ep = Epilepsy(seizure_threshold=0.2)
    assert ep.inhibitory_deficit == pytest.approx(0.8, rel=0.01)
    net = _make_network()
    bg = net.regions["BasalGanglia"]
    initial_dopamine = bg.dopamine
    ep.apply(net)
    # Epilepsy raises excitability (dopamine/excitability proxy)
    assert bg.dopamine >= initial_dopamine


def test_adhd_increases_impulsivity():
    net = _make_network()
    bg = net.regions["BasalGanglia"]
    initial_impulsivity = bg.impulsivity
    adhd = ADHD(severity=0.6)
    adhd.apply(net)
    assert bg.impulsivity > initial_impulsivity
