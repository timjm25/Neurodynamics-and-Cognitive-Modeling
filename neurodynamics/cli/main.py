"""CLI entry point for the Neurodynamics platform."""
import argparse
import json
import sys


def simulate_cmd(args):
    from ..neural_dynamics.neuron_models import (
        LeakyIntegrateAndFire, Izhikevich, HodgkinHuxley
    )
    models = {"lif": LeakyIntegrateAndFire, "izhikevich": Izhikevich,
              "hh": HodgkinHuxley}
    if args.model not in models:
        print(f"Unknown model: {args.model}. Choose from: {list(models)}")
        sys.exit(1)
    model = models[args.model]()
    result = model.simulate(args.duration, args.current)
    print(json.dumps({
        "model": args.model,
        "duration_ms": args.duration,
        "n_spikes": len(result.spike_times),
        "spike_times_ms": result.spike_times.tolist()[:10],
    }, indent=2))


def analyze_cmd(args):
    import numpy as np
    from ..dynamics.information import shannon_entropy
    from ..dynamics.chaos import lyapunov_spectrum
    data = np.array(json.loads(args.data))
    if args.analysis == "entropy":
        hist, _ = np.histogram(data, bins=20)
        p = hist / hist.sum()
        print(f"Shannon entropy: {shannon_entropy(p):.4f} bits")
    elif args.analysis == "lyapunov":
        traj = data.reshape(-1, 1)
        le = lyapunov_spectrum(traj)
        print(f"Lyapunov exponents: {le}")


def serve_cmd(args):
    import uvicorn
    from ..api.rest import app
    uvicorn.run(app, host=args.host, port=args.port)


def main():
    parser = argparse.ArgumentParser(description="Neurodynamics Platform CLI")
    sub = parser.add_subparsers(dest="command")

    sim_parser = sub.add_parser("simulate", help="Simulate a neuron model")
    sim_parser.add_argument("--model", default="lif", help="Model name")
    sim_parser.add_argument("--duration", type=float, default=500.0)
    sim_parser.add_argument("--current", type=float, default=5.0)

    ana_parser = sub.add_parser("analyze", help="Analyze time-series data")
    ana_parser.add_argument("--data", required=True, help="JSON array of values")
    ana_parser.add_argument("--analysis", default="entropy")

    srv_parser = sub.add_parser("serve", help="Start REST API server")
    srv_parser.add_argument("--host", default="0.0.0.0")
    srv_parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()
    if args.command == "simulate":
        simulate_cmd(args)
    elif args.command == "analyze":
        analyze_cmd(args)
    elif args.command == "serve":
        serve_cmd(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
