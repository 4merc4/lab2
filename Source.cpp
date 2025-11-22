#include <algorithm>
#include <execution>
#include <vector>
#include <thread>
#include <iostream>
#include <chrono>
#include <random>
#include <cmath>
#include <iomanip>
#include <functional>

using namespace std;

using clk = chrono::steady_clock;

// median of several runs
double measure(function<void()> fn, int repeats = 7) {
    vector<double> t;
    for (int i = 0; i < repeats; i++) {
        auto t0 = clk::now();
        fn();
        auto t1 = clk::now();
        t.push_back(chrono::duration<double, milli>(t1 - t0).count());
    }
    sort(t.begin(), t.end());
    return t[t.size() / 2];
}

// sequential
template<class Pred>
long long seq_cnt(const vector<int>& a, Pred p) {
    return count_if(a.begin(), a.end(), p);
}

// policy par
template<class Pred>
long long par_cnt(const vector<int>& a, Pred p) {
    return count_if(execution::par, a.begin(), a.end(), p);
}

// policy par_unseq
template<class Pred>
long long par_unseq_cnt(const vector<int>& a, Pred p) {
    return count_if(execution::par_unseq, a.begin(), a.end(), p);
}

// custom parallel K threads
template<class Pred>
long long my_parallel_cnt(const vector<int>& a, Pred p, int K) {

    int n = a.size();
    if (K <= 1) return seq_cnt(a, p);

    vector<long long> partial(K, 0);
    vector<thread> th;
    th.reserve(K);

    for (int k = 0; k < K; k++) {
        int L = (1LL * n * k) / K;
        int R = (1LL * n * (k + 1)) / K;

        th.emplace_back([&, L, R, k]() {
            partial[k] = count_if(a.begin() + L, a.begin() + R, p);
            });
    }
    for (auto& x : th) x.join();

    long long sum = 0;
    for (long long x : partial) sum += x;
    return sum;
}

int main() {

    cout << "CPU threads: " << thread::hardware_concurrency() << "\n\n";

    vector<int> Ns = { 100000, 1000000, 5000000 };
    mt19937_64 rng(123456);

    // light predicate
    auto pred_light = [](int x) {
        return (x & 1) == 0;
        };
    // heavy predicate
    auto pred_heavy = [](int x) {
        double s = 0;
        for (int i = 0; i < 12; i++) s += sqrt(x + i);
        return ((long long)s) & 1;
        };

    struct Pred { string name; function<bool(int)> f; };
    vector<Pred> preds = {
        {"light", pred_light},
        {"heavy", pred_heavy}
    };

    for (int N : Ns) {

        cout << "===========================================================\n";
        cout << "N = " << N << "\n";

        vector<int> a(N);
        uniform_int_distribution<int> d(0, 1'000'000);
        for (int& x : a) x = d(rng);

        for (auto& pp : preds) {
            cout << "\n=== predicate: " << pp.name << " ===\n";

            auto pred = pp.f;

            // Sequential
            long long res_seq = seq_cnt(a, pred);
            double t_seq = measure([&]() { seq_cnt(a, pred); });
            cout << "std::count_if (sequential): " << t_seq << " ms\n";

            // par
            double t_par = -1, t_par_unseq = -1;
            long long res_par = res_seq, res_par_unseq = res_seq;

            try {
                res_par = par_cnt(a, pred);
                t_par = measure([&]() { par_cnt(a, pred); });
                cout << "std::count_if (par):        " << t_par << " ms\n";
            }
            catch (...) {
                cout << "par policy NOT supported\n";
            }

            try {
                res_par_unseq = par_unseq_cnt(a, pred);
                t_par_unseq = measure([&]() { par_unseq_cnt(a, pred); });
                cout << "std::count_if (par_unseq):  " << t_par_unseq << " ms\n";
            }
            catch (...) {
                cout << "par_unseq policy NOT supported\n";
            }

            // Custom K
            cout << "\n--- custom parallel K ----\n";
            int hw = max(1u, thread::hardware_concurrency());
            vector<int> Ks = { 1, 2, 4, 8, 16, 32, 64 };
            for (int k = 2; k <= hw * 2; k *= 2) Ks.push_back(k);

            sort(Ks.begin(), Ks.end());
            Ks.erase(unique(Ks.begin(), Ks.end()), Ks.end());

            double bestT = 1e300;
            int bestK = 1;

            for (int K : Ks) {
                long long res = my_parallel_cnt(a, pred, K);
                double t = measure([&]() { my_parallel_cnt(a, pred, K); });
                if (t < bestT) bestT = t, bestK = K;

                // check correctness
                if (res != res_seq) {
                    cout << "ERROR: incorrect result at K=" << K << "\n";
                }

                cout << "K=" << setw(3) << K << " ? time=" << setw(10) << t << " ms\n";
            }

            cout << "\nBEST K = " << bestK
                << " (speed=" << bestT << " ms, CPU=" << hw
                << ", ratio=" << (double)bestK / hw << ")\n\n";
        }
        cout << "\n\n";
    }

    return 0;
}
