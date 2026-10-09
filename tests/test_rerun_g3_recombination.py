from scripts import rerun_g3_recombination as R


def test_public_routes_preserve_original_tasks_and_need_a_margin():
    rows = {g: {p: {"public": {t: 0.2 for t in R.TASKS}} for p in R.PARENTS}
            for g in R.GENERATIONS}
    for p in R.PARENTS:
        rows["G2"][p]["public"]["doorkey8"] = 0.5
        rows["G3-sp"][p]["public"]["doorkey8"] = 0.56
        rows["G3-trunk"][p]["public"]["unlock"] = 0.24  # below the 0.05 margin
    first, control = R.routes(rows)
    second, control_again = R.routes(rows)
    assert first == second and control == control_again
    for p in R.PARENTS:
        assert all(first[p][t] == "G2" for t in R.TASKS[:4])
        assert first[p]["doorkey8"] == "G3-sp"
        assert first[p]["unlock"] == "G2"
        assert all(control[p][t] == "G2" for t in R.TASKS[:4])
