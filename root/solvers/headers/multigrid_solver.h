#include "iterative_solver.h"
#include "lu_solver.h"

template <typename TMatrix>
class MultiGridSolver {
    public:
        // Constructor with parameters, smoother and base solver.
        MultiGridSolver(IterativeSolver<TMatrix> &smoother,
                        LUSolver<TMatrix> &base_solver,
                        int num_pre_smooth,
                        int num_post_smooth,
                        int num_cycles,
                        int base_elements_per_dim = 2);
        // Destructor.
        ~MultiGridSolver() = default;

        // Solve method - solves A*x = b using multigrid approach.
        // Returns true on success, false otherwise.
        bool solve(TMatrix &A, Vector &x, const Vector &b, 
                   std::size_t num_elements_per_dim);

        void set_base_elements_per_dim(int elements_per_dim) {
            base_elements_per_dim = elements_per_dim;
        }

        void set_num_cycles(int cycles) {
            num_cycles = cycles;
        }

        void set_num_pre_smooth(int pre_smooth) {
            num_pre_smooth = pre_smooth;
        }

        void set_num_post_smooth(int post_smooth) {
            num_post_smooth = post_smooth;
        }

        void set_smoother(IterativeSolver<TMatrix> &s) {
            smoother = s;
        }

        void set_base_solver(LUSolver<TMatrix> &bs) {
            base_solver = bs;
        }

        int get_base_elements_per_dim() const {
            return base_elements_per_dim;
        }

        int get_num_cycles() const {
            return num_cycles;
        }

        int get_num_pre_smooth() const {
            return num_pre_smooth;
        }

        int get_num_post_smooth() const {
            return num_post_smooth;
        }

        IterativeSolver<TMatrix>& get_smoother() const {
            return smoother;
        }

        LUSolver<TMatrix>& get_base_solver() const {
            return base_solver;
        }

    private:
        IterativeSolver<TMatrix>& smoother;
        LUSolver<TMatrix>&        base_solver;
        int                       num_pre_smooth;
        int                       num_post_smooth;
        int                       num_cycles;
        int                       base_elements_per_dim;
};
