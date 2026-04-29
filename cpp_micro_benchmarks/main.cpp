#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

using namespace std;
using namespace std::chrono;

const int N = 30;
const int MAX_L = 8;

uint32_t comb[N + 1][N + 1] = {0};
uint32_t base_offsets[MAX_L + 2] = {0};

void init_combinatorics() {
  for (int i = 0; i <= N; ++i) {
    comb[i][0] = 1;
    for (int j = 1; j <= i; ++j) {
      comb[i][j] = comb[i - 1][j - 1] + comb[i - 1][j];
    }
  }
  base_offsets[0] = 0;
  for (int k = 1; k <= MAX_L + 1; ++k) {
    base_offsets[k] = base_offsets[k - 1] + comb[N][k - 1];
  }
}

inline uint32_t get_dense_index(uint32_t S) {
  int k = __builtin_popcount(S);
  uint32_t idx = base_offsets[k];
  int t = 1;
  while (S > 0) {
    int bit_pos = __builtin_ctz(S); 
    idx += comb[bit_pos][t];
    S &= (S - 1);
    t++;
  }
  return idx;
}

vector<uint32_t> generate_queries(int num_queries, int current_L, int seed) {
  mt19937 gen(seed);
  vector<uint32_t> queries(num_queries);
  for (int i = 0; i < num_queries; ++i) {
    int k = (gen() % current_L) + 1;
    uint32_t mask = 0;
    int bits_set = 0;
    while (bits_set < k) {
      int bit = gen() % N;
      if (!(mask & (1 << bit))) {
        mask |= (1 << bit);
        bits_set++;
      }
    }
    queries[i] = mask;
  }
  return queries;
}

void print_usage(const char *program_name) {
  cerr << "Usage: " << program_name << " <L> <perm> [seed_base]\n";
  cerr << "  L:          subset size limit (1-" << MAX_L << ")\n";
  cerr
      << "  perm:       multiplier for number of queries (num_queries = perm * "
      << N << ")\n";
  cerr << "  seed_base:  base random seed for query generation (optional, "
          "default: 42)\n";
  cerr << "\nExamples:\n";
  cerr << "  " << program_name
       << " 3 200        # L=3, perm=200, seed_base=42\n";
  cerr << "  " << program_name
       << " 5 500 1234   # L=5, perm=500, seed_base=1234\n";
}

int main(int argc, char *argv[]) {
  if (argc < 3 || argc > 4) {
    print_usage(argv[0]);
    return 1;
  }

  int L = atoi(argv[1]);
  int perm = atoi(argv[2]);
  int seed_base = (argc == 4) ? atoi(argv[3]) : 42;

  if (L < 1 || L > MAX_L) {
    cerr << "Error: L must be between 1 and " << MAX_L << ", got " << L << "\n";
    print_usage(argv[0]);
    return 1;
  }
  if (perm <= 0) {
    cerr << "Error: perm must be positive, got " << perm << "\n";
    print_usage(argv[0]);
    return 1;
  }

  init_combinatorics();

  uint32_t total_cache_size = base_offsets[L + 1];
  int num_queries = perm * N;

  double total_time_hash = 0.0;
  double total_time_adapt = 0.0;
  double total_time_direct = 0.0;

  vector<uint32_t> queries = generate_queries(num_queries, L, seed_base);

  auto start_hash = high_resolution_clock::now();
  unordered_map<uint32_t, float> hash_cache;
  hash_cache.reserve(total_cache_size);

  double sum_hash = 0;
  for (uint32_t S : queries) {
    auto it = hash_cache.find(S);
    if (it == hash_cache.end()) {
      hash_cache[S] = 1.0f;
      sum_hash += 1.0f;
    } else {
      sum_hash += it->second;
    }
  }
  auto end_hash = high_resolution_clock::now();
  total_time_hash = duration_cast<duration<double>>(end_hash - start_hash).count();

  auto start_adapt = high_resolution_clock::now();
  unordered_map<uint32_t, float> adapt_cache;
  adapt_cache.reserve(total_cache_size);

  uint32_t size_queries[32] = {0};
  uint32_t size_hits[32] = {0};
  uint32_t adapt_hit = 0, adapt_miss = 0;
  bool size_enabled[32];
  for (int i = 0; i < 32; i++)
    size_enabled[i] = true;

  double sum_adapt = 0;
  for (uint32_t S : queries) {
    int k = __builtin_popcount(S);
    size_queries[k]++;

    if (size_enabled[k]) {
      adapt_hit++;
      auto it = adapt_cache.find(S);
      if (it != adapt_cache.end()) {
        size_hits[k]++;
        sum_adapt += it->second;
      } else {
        adapt_cache[S] = 1.0f;
        sum_adapt += 1.0f;
      }

      if (size_queries[k] % 100 == 0) {
        float hit_rate = (float)size_hits[k] / size_queries[k];
        if (hit_rate < 0.01f) {
          size_enabled[k] = false;
        }
      }
    } else {
      adapt_miss++;
      sum_adapt += 1.0f;
    }
  }
  auto end_adapt = high_resolution_clock::now();
  total_time_adapt =
      duration_cast<duration<double>>(end_adapt - start_adapt).count() *
      (adapt_hit + adapt_miss) / adapt_hit;

  auto start_direct = high_resolution_clock::now();
  size_t alloc_bytes = total_cache_size * sizeof(float);

  float *direct_cache = (float *)mmap(NULL, alloc_bytes, PROT_READ | PROT_WRITE,
                                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (direct_cache == MAP_FAILED)
    return 1;

  madvise(direct_cache, alloc_bytes, MADV_WILLNEED);

  for (size_t i = 0; i < total_cache_size; ++i) {
    direct_cache[i] = -1.0f;
  }

  double sum_direct = 0;
  for (uint32_t S : queries) {
    uint32_t idx = get_dense_index(S);
    if (direct_cache[idx] < 0.0f) {
      direct_cache[idx] = 1.0f;
      sum_direct += 1.0f;
    } else {
      sum_direct += direct_cache[idx];
    }
  }
  auto end_direct = high_resolution_clock::now();
  munmap(direct_cache, alloc_bytes);
  total_time_direct +=
      duration_cast<duration<double>>(end_direct - start_direct).count();

  if (sum_hash != sum_direct || sum_hash != sum_adapt) {
    cerr << "Error: Results mismatch!\n";
    return 1;
  }

  cout << fixed << setprecision(6);
  cout << L << "," << num_queries << "," << total_time_hash << ","
       << total_time_adapt << "," << total_time_direct << "," << setprecision(2)
       << total_time_hash / total_time_direct << "," << setprecision(2)
       << total_time_adapt / total_time_direct << "\n";

  return 0;
}