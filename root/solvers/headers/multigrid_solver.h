#include "iterative_solver.h"
#include "lu_solver.h"
#include <tuple>
#include <map>

// Structure to hold operators and matrix for a single grid level
template <typename TMatrix>
struct GridLevel {
    TMatrix restriction;    // R: fine to coarse
    TMatrix prolongation;   // P: coarse to fine
    TMatrix A_coarse;       // Coarse grid operator
    double  h_minus2;       // 1/h^2 for this level
};

template <typename TMatrix>
class MultiGridSolver {
    public:
        // Constructor with parameters, smoother and base solver.
        MultiGridSolver(IterativeSolver<TMatrix> &smoother,
                        LUSolver<TMatrix> &base_solver);
        // Destructor.
        ~MultiGridSolver() = default;

        // Solve method - solves A*x = b using multigrid approach.
        // Returns tuple of (converged, iterations).
        std::tuple<bool, size_t> solve(const TMatrix &A, Vector &x, const Vector &b, 
                   std::size_t num_elements_per_dim);

        bool init(const Vector &x);

        void set_base_elements_per_dim(int elements_per_dim) {
            base_elements_per_dim = elements_per_dim;
        }

        void set_num_recursions(int recursions) {
            num_recursions = recursions;
        }

        void set_num_pre_smooth(int pre_smooth) {
            num_pre_smooth = pre_smooth;
        }

        void set_num_post_smooth(int post_smooth) {
            num_post_smooth = post_smooth;
        }

        /// Sets the multigrid solver parameters.
        /// 
        /// \param pre_smooth Number of smoothing iterations to perform before recursion.
        /// \param post_smooth Number of smoothing iterations to perform after recursion.
        /// \param recursions Number of recursive multigrid levels to use.
        /// \param base_elements_per_dim Number of base elements per dimension at the coarsest level.
        void set_parameters(int pre_smooth, int post_smooth,
                            int recursions, int base_elements_per_dim) {
            num_pre_smooth = pre_smooth;
            num_post_smooth = post_smooth;
            num_recursions = recursions;
            this->base_elements_per_dim = base_elements_per_dim;
        }

        void set_smoother(IterativeSolver<TMatrix> &s) {
            smoother = s;
        }

        void set_base_solver(LUSolver<TMatrix> &bs) {
            base_solver = bs;
        }

        void set_convergence_params(std::size_t max_iter, double min_def, double min_red) {
            max_iterations = max_iter;
            min_defect = min_def;
            min_reduction = min_red;
        }

        void set_verbose(bool verbose) {bVerbose = verbose; }

        int get_base_elements_per_dim() const {
            return base_elements_per_dim;
        }

        int get_num_recursions() const {
            return num_recursions;
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

        bool get_verbose() const {
            return bVerbose;
        }

    private:
        IterativeSolver<TMatrix>& smoother;
        LUSolver<TMatrix>&        base_solver;
        int                       num_pre_smooth;
        int                       num_post_smooth;
        int                       num_recursions;
        int                       base_elements_per_dim;
        std::size_t               max_iterations;
        double                    min_defect;
        double                    min_reduction;
        bool                      bVerbose;
        
        // Hierarchy cache: maps num_elements_per_dim to grid level data
        mutable std::map<std::size_t, GridLevel<TMatrix>> hierarchy_cache;
        
        // Build the entire multigrid hierarchy for a given finest grid size
        void build_hierarchy(std::size_t finest_elements_per_dim);
        
        // Get or build grid level data for a given grid size
        const GridLevel<TMatrix>& get_grid_level(std::size_t num_elements_per_dim);
        
        // Internal cycle method (single multigrid cycle)
        void cycle(const TMatrix &A, Vector &x, const Vector &b, std::size_t num_elements_per_dim);
};
