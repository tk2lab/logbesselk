import pytest
import numpy as np
import pandas as pd


@pytest.fixture(
    params=["float32", "float64"],
    ids=lambda p: p,
)
def dtype(request):
    dtype = request.param
    return lambda x: np.asarray(x, dtype)


def gen_fixtures(filename):

    def gen_data(filename):
        df = pd.read_csv(filename)
        for i, row in df.iterrows():
            yield row.to_list()

    @pytest.fixture(
        params=gen_data(filename),
        ids=lambda p: f"v{p[0]:+08.3f}x{p[1]:+08.3f}",
    )
    def data(request, dtype):
        v, x, ans = request.param
        yield dtype(v), dtype(x), dtype(ans)

    @pytest.fixture
    def vec_data(dtype):
        v, x, ans = np.array(list(gen_data(filename))).T
        yield dtype(v), dtype(x), dtype(ans)

    return data, vec_data


logk_data, logk_vec_data = gen_fixtures("tests/data/log_k.csv")
ke_data, ke_vec_data = gen_fixtures("tests/data/ke.csv")
#kratio_data, kratio_vec_data = gen_fixtures("mathmatica/kratio.csv")

#dlogkdv_data, dlogkdv_vec_data = gen_fixtures("tests/data/dlogkdv.csv")
dlogkdx_data, dlogkdx_vec_data = gen_fixtures("tests/data/dlogkdx.csv")

logdkdv_data, logdkdv_vec_data = gen_fixtures("tests/data/log_abs_dkdv.csv")
logdkdx_data, logdkdx_vec_data = gen_fixtures("tests/data/log_abs_dkdx.csv")

