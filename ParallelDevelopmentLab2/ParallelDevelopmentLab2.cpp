#include <math.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

const double k = 9.0;
const double a = -k;
const double b = k;

double func(double x) {
	return k * pow(x, k);
}

double anal_integral() {
	if ((int)k % 2 == 1) return 0.0;
	return (2.0 * k * pow(k, k + 1)) / (k + 1);
}

double integrate(double(*func)(double), double a, double b, int n) {
	double h = (b - a) / n;
	double sum = 0.0;
	sum += func(a) + func(b);
	for (int i = 1; i < n; i++)
		sum += 2.0 * func(a + i * h);
	sum *= h / 2.0;
	return sum;
}

double integrate_omp(double(*func)(double), double a, double b, int n) {
	double h = (b - a) / n;
	double sum = 0.0;
#pragma omp parallel 
	{
		int nthreads = omp_get_num_threads();
		int threadid = omp_get_thread_num();
		int items_per_thread = n / nthreads;
		int lb = threadid * items_per_thread;
		int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);
		double sumloc = 0.0;
		if (threadid == 0) sumloc += func(a) / 2.0;
		if (threadid == nthreads - 1) sumloc += func(b) / 2.0;
		for (int i = lb; i <= ub; i++)
			sumloc += func(a + i * h);
#pragma omp atomic
		sum += sumloc;
	}
	sum *= h;
	return sum;
}

double run_serial(int n) {
	double t = omp_get_wtime();
	double res = integrate(func, a, b, n);
	double anal = anal_integral();
	t = omp_get_wtime() - t;

	if (anal == 0.0) {
		printf("Result (serial, n=%d): %.12f; absolute error %.12f; relative error not applicable\n",
			n, res, fabs(res - anal));
	}
	else {
		printf("Result (serial, n=%d): %.12f; absolute error %.12f; relative error %.12f%%\n",
			n, res, fabs(res - anal), fabs((res - anal) / anal) * 100);
	}
	return t;
}

double run_parallel(int n, int num_threads) {
	omp_set_num_threads(num_threads);
	double t = omp_get_wtime();
	double res = integrate_omp(func, a, b, n);
	double anal = anal_integral();
	t = omp_get_wtime() - t;

	if (anal == 0.0) {
		printf("Result (parallel, n=%d, threads=%d): %.12f; absolute error %.12f; relative error not applicable\n",
			n, num_threads, res, fabs(res - anal));
	}
	else {
		printf("Result (parallel, n=%d, threads=%d): %.12f; absolute error %.12f; relative error %.12f%%\n",
			n, num_threads, res, fabs(res - anal), fabs((res - anal) / anal) * 100);
	}
	return t;
}

int main() {
	printf("Integration f(x) = %.0f*x^%.0f on [%.2f, %.2f]\n", k, k, a, b);

	const int nsteps_list[] = { 40000000, 80000000 };
	const int max_threads = 8;

	for (int pass = 0; pass < 2; pass++) {
		int nsteps = nsteps_list[pass];
		printf("\n--- nsteps = %d ---\n", nsteps);

		double tserial = run_serial(nsteps);

		for (int i = 1; i <= max_threads; i++) {
			double tparallel = run_parallel(nsteps, i);
			printf("Execution time (serial, n=%d): %.6f\n", nsteps, tserial);
			printf("Execution time (parallel, n=%d, threads=%d): %.6f\n", nsteps, i, tparallel);
			printf("Speedup (threads=%d): %.2f\n", i, tserial / tparallel);
		}
	}

	return 0;
}
