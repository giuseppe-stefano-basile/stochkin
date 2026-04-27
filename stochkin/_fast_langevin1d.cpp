#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <thread>
#include <vector>

namespace {

uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

double interp_linear(const double* x_grid, const double* values, npy_intp n, double x) {
    if (x <= x_grid[0]) {
        return values[0];
    }
    if (x >= x_grid[n - 1]) {
        return values[n - 1];
    }

    const double* upper = std::upper_bound(x_grid, x_grid + n, x);
    npy_intp hi = static_cast<npy_intp>(upper - x_grid);
    npy_intp lo = hi - 1;
    double x0 = x_grid[lo];
    double x1 = x_grid[hi];
    if (x1 == x0) {
        return values[lo];
    }
    double w = (x - x0) / (x1 - x0);
    return (1.0 - w) * values[lo] + w * values[hi];
}

double reflect_scalar(double x, double lo, double hi) {
    while (x < lo || x > hi) {
        if (x < lo) {
            x = 2.0 * lo - x;
        }
        if (x > hi) {
            x = 2.0 * hi - x;
        }
    }
    return x;
}

int which_basin(double x, const double* x_grid, const int64_t* labels, npy_intp n) {
    if (x < x_grid[0] || x > x_grid[n - 1]) {
        return -1;
    }

    const double* lower = std::lower_bound(x_grid, x_grid + n, x);
    npy_intp idx = static_cast<npy_intp>(lower - x_grid);
    if (idx >= n) {
        idx = n - 1;
    } else if (idx > 0 && std::fabs(x - x_grid[idx - 1]) < std::fabs(x_grid[idx] - x)) {
        idx -= 1;
    }

    int64_t label = labels[idx];
    if (label < 0) {
        return -1;
    }
    return static_cast<int>(label);
}

struct SimulationData {
    const double* x_grid;
    const double* force_grid;
    const double* D_grid;
    const double* gradD_grid;
    const int64_t* labels;
    npy_intp n_grid;
    int n_basins;
    double dt;
    double max_time;
    double beta;
    double lo;
    double hi;
    int boundary_mode;
    int trials_per_basin;
    uint64_t seed;
    std::vector<std::vector<npy_intp> > basin_indices;
    int64_t* targets;
    double* times;
};

void run_range(const SimulationData* data, npy_intp begin, npy_intp end) {
    const int n_steps = static_cast<int>(std::floor(data->max_time / data->dt));
    const double nan = std::numeric_limits<double>::quiet_NaN();

    for (npy_intp global_idx = begin; global_idx < end; ++global_idx) {
        int start_id = static_cast<int>(global_idx / data->trials_per_basin);
        data->targets[global_idx] = -1;
        data->times[global_idx] = nan;

        if (start_id < 0 || start_id >= data->n_basins) {
            continue;
        }
        const std::vector<npy_intp>& starts = data->basin_indices[start_id];
        if (starts.empty()) {
            continue;
        }

        uint64_t trial_seed = splitmix64(
            data->seed ^ (static_cast<uint64_t>(global_idx) * 0xd1b54a32d192ed03ULL)
        );
        std::mt19937_64 rng(trial_seed);
        std::uniform_int_distribution<npy_intp> start_dist(0, starts.size() - 1);
        std::normal_distribution<double> normal(0.0, 1.0);

        double x = data->x_grid[starts[start_dist(rng)]];
        if (which_basin(x, data->x_grid, data->labels, data->n_grid) != start_id) {
            continue;
        }

        double t = 0.0;
        for (int step = 0; step < n_steps; ++step) {
            double force = interp_linear(data->x_grid, data->force_grid, data->n_grid, x);
            double D = interp_linear(data->x_grid, data->D_grid, data->n_grid, x);
            double gradD = interp_linear(data->x_grid, data->gradD_grid, data->n_grid, x);
            if (!(D >= 0.0) || !std::isfinite(D)) {
                break;
            }

            double drift = data->beta * D * force + gradD;
            double noise = std::sqrt(2.0 * D * data->dt) * normal(rng);
            x += drift * data->dt + noise;

            if (data->boundary_mode == 0) {
                x = reflect_scalar(x, data->lo, data->hi);
            } else {
                x = std::min(std::max(x, data->lo), data->hi);
            }

            t += data->dt;
            int basin = which_basin(x, data->x_grid, data->labels, data->n_grid);
            if (basin >= 0 && basin != start_id) {
                data->targets[global_idx] = static_cast<int64_t>(basin);
                data->times[global_idx] = t;
                break;
            }
        }
    }
}

PyObject* simulate_first_exit_network(PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject* x_obj = nullptr;
    PyObject* force_obj = nullptr;
    PyObject* D_obj = nullptr;
    PyObject* gradD_obj = nullptr;
    PyObject* labels_obj = nullptr;
    int n_basins = 0;
    double dt = 0.0;
    double max_time = 0.0;
    double beta = 0.0;
    double lo = 0.0;
    double hi = 0.0;
    int boundary_mode = 0;
    int trials_per_basin = 0;
    unsigned long long seed = 0;
    int n_threads = 1;

    static const char* kwlist[] = {
        "x_grid",
        "force_grid",
        "D_grid",
        "gradD_grid",
        "labels",
        "n_basins",
        "dt",
        "max_time",
        "beta",
        "lo",
        "hi",
        "boundary_mode",
        "trials_per_basin",
        "seed",
        "n_threads",
        nullptr,
    };

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OOOOOidddddiiKi",
            const_cast<char**>(kwlist),
            &x_obj,
            &force_obj,
            &D_obj,
            &gradD_obj,
            &labels_obj,
            &n_basins,
            &dt,
            &max_time,
            &beta,
            &lo,
            &hi,
            &boundary_mode,
            &trials_per_basin,
            &seed,
            &n_threads)) {
        return nullptr;
    }

    PyArrayObject* x_arr = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* force_arr = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(force_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* D_arr = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(D_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* gradD_arr = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(gradD_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* labels_arr = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(labels_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY));

    if (!x_arr || !force_arr || !D_arr || !gradD_arr || !labels_arr) {
        Py_XDECREF(x_arr);
        Py_XDECREF(force_arr);
        Py_XDECREF(D_arr);
        Py_XDECREF(gradD_arr);
        Py_XDECREF(labels_arr);
        return nullptr;
    }

    npy_intp n_grid = PyArray_SIZE(x_arr);
    bool valid = (
        PyArray_NDIM(x_arr) == 1 &&
        PyArray_NDIM(force_arr) == 1 &&
        PyArray_NDIM(D_arr) == 1 &&
        PyArray_NDIM(gradD_arr) == 1 &&
        PyArray_NDIM(labels_arr) == 1 &&
        PyArray_SIZE(force_arr) == n_grid &&
        PyArray_SIZE(D_arr) == n_grid &&
        PyArray_SIZE(gradD_arr) == n_grid &&
        PyArray_SIZE(labels_arr) == n_grid &&
        n_grid >= 2 &&
        n_basins > 0 &&
        trials_per_basin > 0 &&
        dt > 0.0 &&
        max_time >= 0.0 &&
        beta > 0.0 &&
        lo < hi &&
        (boundary_mode == 0 || boundary_mode == 1)
    );
    if (!valid) {
        Py_DECREF(x_arr);
        Py_DECREF(force_arr);
        Py_DECREF(D_arr);
        Py_DECREF(gradD_arr);
        Py_DECREF(labels_arr);
        PyErr_SetString(PyExc_ValueError, "invalid fast Langevin input arrays or parameters");
        return nullptr;
    }

    npy_intp n_total = static_cast<npy_intp>(n_basins) * static_cast<npy_intp>(trials_per_basin);
    npy_intp dims[1] = {n_total};
    PyArrayObject* targets_arr = reinterpret_cast<PyArrayObject*>(
        PyArray_SimpleNew(1, dims, NPY_INT64));
    PyArrayObject* times_arr = reinterpret_cast<PyArrayObject*>(
        PyArray_SimpleNew(1, dims, NPY_DOUBLE));
    if (!targets_arr || !times_arr) {
        Py_XDECREF(targets_arr);
        Py_XDECREF(times_arr);
        Py_DECREF(x_arr);
        Py_DECREF(force_arr);
        Py_DECREF(D_arr);
        Py_DECREF(gradD_arr);
        Py_DECREF(labels_arr);
        return nullptr;
    }

    const double* x_grid = static_cast<const double*>(PyArray_DATA(x_arr));
    const double* force_grid = static_cast<const double*>(PyArray_DATA(force_arr));
    const double* D_grid = static_cast<const double*>(PyArray_DATA(D_arr));
    const double* gradD_grid = static_cast<const double*>(PyArray_DATA(gradD_arr));
    const int64_t* labels = static_cast<const int64_t*>(PyArray_DATA(labels_arr));

    std::vector<std::vector<npy_intp> > basin_indices(static_cast<size_t>(n_basins));
    for (npy_intp i = 0; i < n_grid; ++i) {
        int64_t label = labels[i];
        if (label >= 0 && label < n_basins) {
            basin_indices[static_cast<size_t>(label)].push_back(i);
        }
    }

    int64_t* targets = static_cast<int64_t*>(PyArray_DATA(targets_arr));
    double* times = static_cast<double*>(PyArray_DATA(times_arr));

    if (n_threads < 1) {
        n_threads = 1;
    }
    unsigned int hardware = std::thread::hardware_concurrency();
    if (hardware > 0) {
        n_threads = std::min(n_threads, static_cast<int>(hardware));
    }
    n_threads = std::min<npy_intp>(n_threads, std::max<npy_intp>(1, n_total));

    SimulationData data;
    data.x_grid = x_grid;
    data.force_grid = force_grid;
    data.D_grid = D_grid;
    data.gradD_grid = gradD_grid;
    data.labels = labels;
    data.n_grid = n_grid;
    data.n_basins = n_basins;
    data.dt = dt;
    data.max_time = max_time;
    data.beta = beta;
    data.lo = lo;
    data.hi = hi;
    data.boundary_mode = boundary_mode;
    data.trials_per_basin = trials_per_basin;
    data.seed = static_cast<uint64_t>(seed);
    data.basin_indices = basin_indices;
    data.targets = targets;
    data.times = times;

    Py_BEGIN_ALLOW_THREADS
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(n_threads));
    for (int thread_id = 0; thread_id < n_threads; ++thread_id) {
        npy_intp begin = (n_total * thread_id) / n_threads;
        npy_intp end = (n_total * (thread_id + 1)) / n_threads;
        workers.emplace_back(run_range, &data, begin, end);
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
    Py_END_ALLOW_THREADS

    Py_DECREF(x_arr);
    Py_DECREF(force_arr);
    Py_DECREF(D_arr);
    Py_DECREF(gradD_arr);
    Py_DECREF(labels_arr);

    PyObject* result = PyTuple_New(2);
    if (!result) {
        Py_DECREF(targets_arr);
        Py_DECREF(times_arr);
        return nullptr;
    }
    PyTuple_SET_ITEM(result, 0, reinterpret_cast<PyObject*>(targets_arr));
    PyTuple_SET_ITEM(result, 1, reinterpret_cast<PyObject*>(times_arr));
    return result;
}

PyMethodDef methods[] = {
    {
        "simulate_first_exit_network",
        reinterpret_cast<PyCFunction>(simulate_first_exit_network),
        METH_VARARGS | METH_KEYWORDS,
        "Run fast 1D overdamped first-exit trajectories.",
    },
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_fast_langevin1d",
    "Compiled helpers for fast 1D Langevin first-exit simulations.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__fast_langevin1d(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
