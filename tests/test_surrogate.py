from pathlib import Path
import sys
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[1]))

from sampling import define_random_variables, generate_ccd_samples
from fsi import run_coupled_simulation
from surrogate import MultiRSM
from reliability import create_limit_state, iterate_until_convergence


def test_multi_rsm_fit():
    mu, std, dists = define_random_variables()
    samples = generate_ccd_samples(mu, std, n_center=1)
    sims = [run_coupled_simulation(None, s, seed=i, use_fsi=False) for i, s in enumerate(samples.to_dict('records'))]
    df = pd.DataFrame(sims)
    rsm = MultiRSM(pop_size=0, n_gen=0)
    rsm.fit_all(samples, df)
    preds = rsm.predict('Ucr', samples)
    assert preds.shape[0] == len(samples)


def test_iterate_convergence():
    mu, std, dists = define_random_variables()
    g = create_limit_state("Ucr - U10")
    rsm, history = iterate_until_convergence(
        None,
        mu,
        std,
        dists,
        g_func=g,
        max_iter=3,
        pop_size=0,
        n_gen=0,
        use_fsi=False,
    )
    assert len(history) <= 3
