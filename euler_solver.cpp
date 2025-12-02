#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <limits>
#include <omp.h> // Include OpenMP header

// Ghost cell layers
#define NGHOST 2

// Mathematical constants
const double PI = 3.14159;
const double GAMMA = 1.4;
const double EPS = 1e-12;

// Struct to hold conservative variables
struct ConservativeVars {
    double rho;   // density
    double rhou;  // x-momentum
    double rhov;  // y-momentum
    double rhoE;  // total energy
    
    ConservativeVars() : rho(0), rhou(0), rhov(0), rhoE(0) {}

    // --- Operator overloads (needed for RK4 math) ---
    ConservativeVars operator+(const ConservativeVars& b) const {
        ConservativeVars result;
        result.rho = rho + b.rho;
        result.rhou = rhou + b.rhou;
        result.rhov = rhov + b.rhov;
        result.rhoE = rhoE + b.rhoE;
        return result;
    }
    ConservativeVars operator-(const ConservativeVars& b) const {
        ConservativeVars result;
        result.rho = rho - b.rho;
        result.rhou = rhou - b.rhou;
        result.rhov = rhov - b.rhov;
        result.rhoE = rhoE - b.rhoE;
        return result;
    }
    ConservativeVars& operator+=(const ConservativeVars& b) {
        rho += b.rho; rhou += b.rhou; rhov += b.rhov; rhoE += b.rhoE;
        return *this;
    }
    ConservativeVars& operator-=(const ConservativeVars& b) {
        rho -= b.rho; rhou -= b.rhou; rhov -= b.rhov; rhoE -= b.rhoE;
        return *this;
    }
    ConservativeVars operator*(double s) const {
        ConservativeVars result;
        result.rho = rho * s;
        result.rhou = rhou * s;
        result.rhov = rhov * s;
        result.rhoE = rhoE * s;
        return result;
    }
};

// Non-member operator for scalar * ConservativeVars
inline ConservativeVars operator*(double s, const ConservativeVars& v) {
    return v * s;
}


// Struct to hold primitive variables
struct PrimitiveVars {
    double rho;
    double u;
    double v;
    double p;
    
    PrimitiveVars() : rho(0), u(0), v(0), p(0) {}
};


struct Mesh {
    int ni, nj;       // nodes in i and j directions
    int nic, njc;     // cells in i and j directions
    
    std::vector<double> x, y;       // node coordinates [node_stride = ni+2*NGHOST]
    std::vector<double> xc, yc;     // cell center coordinates [cell_stride = nic+2*NGHOST]
    std::vector<double> vol;      // cell volumes
    
    // Face data stored separately for i and j directions
    std::vector<double> Sx_i, Sy_i; // i-direction face normals
    std::vector<double> Sx_j, Sy_j; // j-direction face normals
    
    int node_stride() const { return ni + 2*NGHOST; }
    int cell_stride() const { return nic + 2*NGHOST; }
};

// Class for the Euler solver
class EulerSolver {
private:
    Mesh mesh;
    double M_inf, alpha;
    double beta_PG; // NEW: Prandtl-Glauert factor sqrt(1 - M_inf^2)
    double rho_inf, u_inf, v_inf, p_inf, a_inf, E_inf, T_inf;
    
    std::vector<ConservativeVars> U;      // Conservative variables (cell-centered)
    std::vector<ConservativeVars> U_old;  // Old time level
    std::vector<ConservativeVars> R;      // Residuals
    std::vector<ConservativeVars> R_temp; // Buffer for residual smoothing
    std::vector<double> dt_local;       // Local time step (set to global)
    
    double CFL;
    int max_iter;
    double conv_tol;
    
    // Residual Smoothing Parameters
    bool use_residual_smoothing;
    double res_smoothing_coeff;

    std::vector<double> residual_history;
    
public:
    EulerSolver(const std::string& mesh_file, double Mach, double alpha_deg, double CFL_num = 0.5)
        : M_inf(Mach), alpha(alpha_deg * PI / 180.0), CFL(CFL_num), 
          max_iter(10000), conv_tol(1e-7),
          use_residual_smoothing(true), res_smoothing_coeff(2.0) { 
        
        // --- NEW: Calculate Prandtl-Glauert factor (beta) ---
        if (M_inf < 1.0) {
            beta_PG = std::sqrt(1.0 - M_inf * M_inf);
        } else {
            beta_PG = 0.0;
            std::cerr << "Warning: Prandtl-Glauert correction is not applicable for M >= 1." << std::endl;
        }
        // --- END NEW ---

        readMesh(mesh_file);
        computeMetrics();
        initializeFlow();
    }
    
    void readMesh(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open mesh file " << filename << std::endl;
            exit(1);
        }
        
        int nblock;
        file >> nblock >> mesh.ni >> mesh.nj;
        
        std::cout << "Reading mesh: " << mesh.ni << " x " << mesh.nj << " nodes" << std::endl;
        
        mesh.nic = mesh.ni - 1;
        mesh.njc = mesh.nj - 1;
        
        // Allocate node arrays 
        int node_size = mesh.node_stride() * (mesh.nj + 2*NGHOST);
        mesh.x.resize(node_size, 0.0);
        mesh.y.resize(node_size, 0.0);
        
        // Read nodes into interior region only (Serial I/O)
        for (int jj = 0; jj < mesh.nj; jj++) {
            for (int ii = 0; ii < mesh.ni; ii++) {
                double coord;
                file >> coord;
                int index = (jj + NGHOST) * mesh.node_stride() + (ii + NGHOST);
                mesh.x[index] = coord;
            }
        }
        
        for (int jj = 0; jj < mesh.nj; jj++) {
            for (int ii = 0; ii < mesh.ni; ii++) {
                double coord;
                file >> coord;
                int index = (jj + NGHOST) * mesh.node_stride() + (ii + NGHOST);
                mesh.y[index] = coord;
            }
        }
        
        file.close();
        std::cout << "Mesh read successfully" << std::endl;
    }
    
    void computeMetrics() {
        int cell_inci = mesh.cell_stride();
        int node_inci = mesh.node_stride();
        
        // Allocate cell arrays
        int cell_size = cell_inci * (mesh.njc + 2*NGHOST);
        mesh.xc.resize(cell_size, 0.0);
        mesh.yc.resize(cell_size, 0.0);
        mesh.vol.resize(cell_size, 0.0);
        
        // Allocate face arrays
        mesh.Sx_i.resize(cell_inci * (mesh.njc + 2*NGHOST), 0.0);
        mesh.Sy_i.resize(cell_inci * (mesh.njc + 2*NGHOST), 0.0);
        mesh.Sx_j.resize(cell_inci * (mesh.njc + 2*NGHOST + 1), 0.0);
        mesh.Sy_j.resize(cell_inci * (mesh.njc + 2*NGHOST + 1), 0.0);
        
        // Compute cell centers and volumes for interior cells
        #pragma omp parallel for
        for (int jj = NGHOST; jj < mesh.njc + NGHOST; jj++) {
            for (int ii = NGHOST; ii < mesh.nic + NGHOST; ii++) {
                int icell = jj * cell_inci + ii;
                
                // Four corner nodes
                int inode1 = jj * node_inci + ii;
                int inode2 = (jj+1) * node_inci + ii;
                int inode3 = jj * node_inci + (ii+1);
                int inode4 = (jj+1) * node_inci + (ii+1);
                
                // Cell center
                mesh.xc[icell] = 0.25 * (mesh.x[inode1] + mesh.x[inode2] + 
                                         mesh.x[inode3] + mesh.x[inode4]);
                mesh.yc[icell] = 0.25 * (mesh.y[inode1] + mesh.y[inode2] + 
                                         mesh.y[inode3] + mesh.y[inode4]);
                
                // Cell volume using shoelace formula
                double x1 = mesh.x[inode1], y1 = mesh.y[inode1];
                double x2 = mesh.x[inode3], y2 = mesh.y[inode3];
                double x3 = mesh.x[inode4], y3 = mesh.y[inode4];
                double x4 = mesh.x[inode2], y4 = mesh.y[inode2];
                
                mesh.vol[icell] = 0.5 * std::abs(
                    (x1*y2 - x2*y1) + (x2*y3 - x3*y2) + 
                    (x3*y4 - x4*y3) + (x4*y1 - x1*y4)
                );
                
                if (mesh.vol[icell] < EPS) {
                    mesh.vol[icell] = EPS;
                }
            }
        }
        
        // Compute face normals in i-direction
        #pragma omp parallel for
        for (int jj = NGHOST; jj < mesh.njc + NGHOST; jj++) {
            for (int ii = NGHOST; ii <= mesh.nic + NGHOST; ii++) {
                int iface = jj * cell_inci + ii;
                
                int n1 = jj * node_inci + ii;
                int n2 = (jj+1) * node_inci + ii;
                
                double dx = mesh.x[n2] - mesh.x[n1];
                double dy = mesh.y[n2] - mesh.y[n1];
                
                mesh.Sx_i[iface] = dy;   // Outward normal
                mesh.Sy_i[iface] = -dx;
            }
        }
        
        // Compute face normals in j-direction
        #pragma omp parallel for
        for (int jj = NGHOST; jj <= mesh.njc + NGHOST; jj++) {
            for (int ii = NGHOST; ii < mesh.nic + NGHOST; ii++) {
                int iface = jj * cell_inci + ii;
                
                int n1 = jj * node_inci + ii;
                int n2 = jj * node_inci + (ii+1);
                
                double dx = mesh.x[n2] - mesh.x[n1];
                double dy = mesh.y[n2] - mesh.y[n1];
                
                mesh.Sx_j[iface] = -dy;  // Outward normal
                mesh.Sy_j[iface] = dx;
            }
        }
        
        // Fill ghost cell metrics using extrapolation 
        fillGhostCellMetrics();
        
        std::cout << "Mesh metrics computed" << std::endl;
    }
    
    void fillGhostCellMetrics() {
        int cell_inci = mesh.cell_stride();
        
        // j-min ghost cells (extrapolation)
        #pragma omp parallel for
        for (int ii = NGHOST; ii < mesh.nic + NGHOST; ii++) {
            for (int g = 1; g <= NGHOST; g++) {
                int jj = NGHOST - g;
                int jj_p1 = NGHOST - g + 1;
                int jj_p2 = NGHOST - g + 2;
                
                int icell = jj * cell_inci + ii;
                int icellp1 = jj_p1 * cell_inci + ii;
                int icellp2 = jj_p2 * cell_inci + ii;
                
                mesh.xc[icell] = mesh.xc[icellp1] + (mesh.xc[icellp1] - mesh.xc[icellp2]);
                mesh.yc[icell] = mesh.yc[icellp1] + (mesh.yc[icellp1] - mesh.yc[icellp2]);
                mesh.vol[icell] = mesh.vol[icellp1];
            }
        }
        
        // j-max ghost cells (extrapolation)
        #pragma omp parallel for
        for (int ii = NGHOST; ii < mesh.nic + NGHOST; ii++) {
            for (int g = 0; g < NGHOST; g++) {
                int jj = mesh.njc + NGHOST + g;
                int jj_m1 = mesh.njc + NGHOST - 1 + g;
                int jj_m2 = mesh.njc + NGHOST - 2 + g;
                
                int icell = jj * cell_inci + ii;
                int icellm1 = jj_m1 * cell_inci + ii;
                int icellm2 = jj_m2 * cell_inci + ii;
                
                mesh.xc[icell] = mesh.xc[icellm1] + (mesh.xc[icellm1] - mesh.xc[icellm2]);
                mesh.yc[icell] = mesh.yc[icellm1] + (mesh.yc[icellm1] - mesh.yc[icellm2]);
                mesh.vol[icell] = mesh.vol[icellm1];
            }
        }
        
        // i-min and i-max: periodic
        #pragma omp parallel for
        for (int jj = 0; jj < mesh.njc + 2*NGHOST; jj++) {
            for (int g = 1; g <= NGHOST; g++) {
                int ii_left_ghost = NGHOST - g;
                int ii_right_ghost = mesh.nic + NGHOST - 1 + g;
                int ii_from_right = mesh.nic + NGHOST - g;
                int ii_from_left = NGHOST - 1 + g;

                int icell_left_ghost = jj * cell_inci + ii_left_ghost;
                int icell_right_ghost = jj * cell_inci + ii_right_ghost;
                int icell_from_right = jj * cell_inci + ii_from_right;
                int icell_from_left = jj * cell_inci + ii_from_left;

                mesh.xc[icell_left_ghost] = mesh.xc[icell_from_right];
                mesh.yc[icell_left_ghost] = mesh.yc[icell_from_right];
                mesh.vol[icell_left_ghost] = mesh.vol[icell_from_right];

                mesh.xc[icell_right_ghost] = mesh.xc[icell_from_left];
                mesh.yc[icell_right_ghost] = mesh.yc[icell_from_left];
                mesh.vol[icell_right_ghost] = mesh.vol[icell_from_left];
            }
        }
    }
    
    void initializeFlow() {
        T_inf = 1.0;
        rho_inf = 1.0;
        p_inf = rho_inf * T_inf / GAMMA;
        a_inf = std::sqrt(GAMMA * p_inf / rho_inf);
        u_inf = M_inf * a_inf * std::cos(alpha);
        v_inf = M_inf * a_inf * std::sin(alpha);
        
        double V2 = u_inf * u_inf + v_inf * v_inf;
        E_inf = p_inf / (rho_inf * (GAMMA - 1.0)) + 0.5 * V2;
        
        int ncells = mesh.cell_stride() * (mesh.njc + 2*NGHOST);
        
        U.resize(ncells);
        U_old.resize(ncells);
        R.resize(ncells);
        R_temp.resize(ncells); // Resize new buffer
        dt_local.resize(ncells, 0.0);
        
        // Initialize with freestream
        #pragma omp parallel for
        for (int i = 0; i < ncells; i++) {
            U[i].rho = rho_inf;
            U[i].rhou = rho_inf * u_inf;
            U[i].rhov = rho_inf * v_inf;
            U[i].rhoE = rho_inf * E_inf;
        }
        
        std::cout << "Flow initialized: M=" << M_inf << ", alpha=" << alpha*180/PI 
                  << " deg, p_inf=" << p_inf << ", a_inf=" << a_inf << std::endl;
    }
    
    PrimitiveVars conservative2Primitive(const ConservativeVars& Uc) const {
        PrimitiveVars W;
        W.rho = std::max(Uc.rho, EPS);
        W.u = Uc.rhou / W.rho;
        W.v = Uc.rhov / W.rho;
        double E = Uc.rhoE / W.rho;
        double V2 = W.u * W.u + W.v * W.v;
        W.p = (GAMMA - 1.0) * W.rho * (E - 0.5 * V2);
        W.p = std::max(W.p, EPS * p_inf);
        return W;
    }
    
    ConservativeVars primitive2Conservative(const PrimitiveVars& W) const {
        ConservativeVars Uc;
        Uc.rho = W.rho;
        Uc.rhou = W.rho * W.u;
        Uc.rhov = W.rho * W.v;
        double V2 = W.u * W.u + W.v * W.v;
        Uc.rhoE = W.rho * (W.p / (W.rho * (GAMMA - 1.0)) + 0.5 * V2);
        return Uc;
    }
    
    void applyBoundaryConditions() {
        int cell_inci = mesh.cell_stride();
        int njc_total = mesh.njc + 2*NGHOST;
        
        // STEP 1: Periodic BC in i-direction (FIRST for O-grid!)
        #pragma omp parallel for
        for (int jj = 0; jj < njc_total; jj++) {
            for (int g = 1; g <= NGHOST; g++) {
                // Left ghost ← Right interior
                U[jj * cell_inci + (NGHOST - g)] = U[jj * cell_inci + (mesh.nic + NGHOST - g)];
                
                // Right ghost ← Left interior
                U[jj * cell_inci + (mesh.nic + NGHOST - 1 + g)] = U[jj * cell_inci + (NGHOST - 1 + g)];
            }
        }
        
        // STEP 2: Wall BC at j-min (slip wall)
        #pragma omp parallel for
        for (int ii = 0; ii < cell_inci; ii++) {  // All i including ghosts
            for (int g = 1; g <= NGHOST; g++) {
                int jj_wall = NGHOST - g;
                int jj_int = NGHOST + g - 1;
                
                int idx_wall = jj_wall * cell_inci + ii;
                int idx_int = jj_int * cell_inci + ii;
                int face_idx = NGHOST * cell_inci + ii;
                
                PrimitiveVars W_int = conservative2Primitive(U[idx_int]);
                
                // Get wall normal
                double nx = -mesh.Sx_j[face_idx];
                double ny = -mesh.Sy_j[face_idx];
                double mag = std::sqrt(nx*nx + ny*ny) + EPS;
                nx /= mag;
                ny /= mag;
                
                // Reflect velocity
                double vn = W_int.u * nx + W_int.v * ny;
                
                PrimitiveVars W_wall;
                W_wall.rho = W_int.rho;
                W_wall.u = W_int.u - 2.0 * vn * nx;
                W_wall.v = W_int.v - 2.0 * vn * ny;
                W_wall.p = W_int.p;
                
                U[idx_wall] = primitive2Conservative(W_wall);
            }
        }
        
        // STEP 3: Farfield BC at j-max
        #pragma omp parallel for
        for (int ii = 0; ii < cell_inci; ii++) {  // All i including ghosts
            for (int g = 0; g < NGHOST; g++) {
                int jj_ghost = mesh.njc + NGHOST + g;
                int jj_int = mesh.njc + NGHOST - 1;
                
                int idx_ghost = jj_ghost * cell_inci + ii;
                int idx_int = jj_int * cell_inci + ii;
                
                PrimitiveVars W_int = conservative2Primitive(U[idx_int]);
                
                // Subsonic: extrapolate except pressure
                PrimitiveVars W_ghost;
                W_ghost.rho = W_int.rho;
                W_ghost.u = W_int.u;
                W_ghost.v = W_int.v;
                W_ghost.p = p_inf;
                
                U[idx_ghost] = primitive2Conservative(W_ghost);
            }
        }
    }
    
    // --- Global Time Stepping (for stability) ---
    void computeTimeStep() {
        int cell_inci = mesh.cell_stride();
        double dt_min = 1e30; 
        
        #pragma omp parallel for reduction(min:dt_min)
        for (int jj = NGHOST; jj < mesh.njc + NGHOST; jj++) {
            for (int ii = NGHOST; ii < mesh.nic + NGHOST; ii++) {
                int idx = jj * cell_inci + ii;
                
                PrimitiveVars W = conservative2Primitive(U[idx]);
                double a = std::sqrt(GAMMA * W.p / W.rho);
                double V_mag = std::sqrt(W.u*W.u + W.v*W.v);
                
                double lambda_max = V_mag + a;
                double h = std::sqrt(mesh.vol[idx]);
                
                // Calculate local dt first
                double dt_val = CFL * h / (lambda_max + EPS);
                dt_local[idx] = dt_val;
                
                // Find the minimum dt across the whole domain
                if (dt_val < dt_min) {
                    dt_min = dt_val;
                }
            }
        }
        
        // --- Global Time Step ---
        // Overwrite all local dt values with the single global minimum
        #pragma omp parallel for
        for (int jj = NGHOST; jj < mesh.njc + NGHOST; jj++) {
            for (int ii = NGHOST; ii < mesh.nic + NGHOST; ii++) {
                dt_local[jj * cell_inci + ii] = dt_min;
            }
        }
    }
    
    void computeResiduals() {
        int cell_inci = mesh.cell_stride();
        
        // Reset residuals
        #pragma omp parallel for
        for (size_t i = 0; i < R.size(); i++) {
            R[i].rho = R[i].rhou = R[i].rhov = R[i].rhoE = 0.0;
        }
        
        // i-direction fluxes
        #pragma omp parallel for
        for (int jj = NGHOST; jj < mesh.njc + NGHOST; jj++) {
            for (int ii = NGHOST; ii <= mesh.nic + NGHOST; ii++) {
                int idx_L = jj * cell_inci + (ii-1);
                int idx_R = jj * cell_inci + ii;
                int face_idx = jj * cell_inci + ii;
                
                double Sx = mesh.Sx_i[face_idx];
                double Sy = mesh.Sy_i[face_idx];
                
                auto flux = computeRoeFlux(U[idx_L], U[idx_R], Sx, Sy);
                
                if (ii-1 >= NGHOST && ii-1 < mesh.nic + NGHOST) {
                    // This is thread-safe because each thread 'jj' only writes
                    // to its own 'jj' row.
                    R[idx_L].rho += flux.rho;
                    R[idx_L].rhou += flux.rhou;
                    R[idx_L].rhov += flux.rhov;
                    R[idx_L].rhoE += flux.rhoE;
                }
                
                if (ii >= NGHOST && ii < mesh.nic + NGHOST) {
                    // This is thread-safe for the same reason.
                    R[idx_R].rho -= flux.rho;
                    R[idx_R].rhou -= flux.rhou;
                    R[idx_R].rhov -= flux.rhov;
                    R[idx_R].rhoE -= flux.rhoE;
                }
            }
        }
        
        // j-direction fluxes
        // --- LOOP ORDER SWAPPED for parallel safety ---
        #pragma omp parallel for
        for (int ii = NGHOST; ii < mesh.nic + NGHOST; ii++) {
            for (int jj = NGHOST; jj <= mesh.njc + NGHOST; jj++) {
                int idx_L = (jj-1) * cell_inci + ii;
                int idx_R = jj * cell_inci + ii;
                int face_idx = jj * cell_inci + ii;
                
                double Sx = mesh.Sx_j[face_idx];
                double Sy = mesh.Sy_j[face_idx];
                
                auto flux = computeRoeFlux(U[idx_L], U[idx_R], Sx, Sy);
                
                if (jj-1 >= NGHOST && jj-1 < mesh.njc + NGHOST) {
                    // This is thread-safe because each thread 'ii' only writes
                    // to its own 'ii' column.
                    R[idx_L].rho += flux.rho;
                    R[idx_L].rhou += flux.rhou;
                    R[idx_L].rhov += flux.rhov;
                    R[idx_L].rhoE += flux.rhoE;
                }
                
                if (jj >= NGHOST && jj < mesh.njc + NGHOST) {
                    // This is thread-safe for the same reason.
                    R[idx_R].rho -= flux.rho;
                    R[idx_R].rhou -= flux.rhou;
                    R[idx_R].rhov -= flux.rhov;
                    R[idx_R].rhoE -= flux.rhoE;
                }
            }
        }
    }

    // --- Tridiagonal Solver (Thomas Algorithm) ---
    void tdma_solve(const std::vector<double>& a, const std::vector<double>& b,
                     const std::vector<double>& c, const std::vector<double>& d,
                     std::vector<double>& x, int n) {
        
        std::vector<double> c_prime(n);
        std::vector<double> d_prime(n);

        // Forward sweep
        c_prime[0] = c[0] / b[0];
        d_prime[0] = d[0] / b[0];

        for (int i = 1; i < n; i++) {
            double m = b[i] - a[i] * c_prime[i-1];
            if (std::abs(m) < EPS) m = EPS; // Avoid division by zero
            c_prime[i] = c[i] / m;
            d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / m;
        }

        // Backward substitution
        x[n-1] = d_prime[n-1];
        for (int i = n - 2; i >= 0; i--) {
            x[i] = d_prime[i] - c_prime[i] * x[i+1];
        }
    }

    // --- Implicit Residual Smoothing ---
    void smoothResiduals() {
        if (!use_residual_smoothing || res_smoothing_coeff < EPS) {
            return; // Skip if smoothing is off
        }

        int cell_inci = mesh.cell_stride();
        int ni = mesh.nic;
        int nj = mesh.njc;
        double epsilon = res_smoothing_coeff;
        
        R_temp = R; // Use R_temp as the destination for the i-sweep

        // --- i-sweep (Horizontal) ---
        #pragma omp parallel for
        for (int jj = NGHOST; jj < nj + NGHOST; jj++) {
            // --- These vectors must be thread-private ---
            int max_dim = ni;
            std::vector<double> a(max_dim);
            std::vector<double> b(max_dim);
            std::vector<double> c(max_dim);
            std::vector<double> d(max_dim);
            std::vector<double> x(max_dim);
            // ------------------------------------------

            // Setup TDMA system for this j-line
            for (int i = 0; i < ni; i++) {
                a[i] = -epsilon;
                b[i] = 1.0 + 2.0 * epsilon;
                c[i] = -epsilon;
            }
            // Neumann BCs at ends (zero-gradient for residual)
            a[0] = 0.0;
            c[ni-1] = 0.0;
            b[0] = 1.0 + epsilon;
            b[ni-1] = 1.0 + epsilon;

            // Solve for rho
            for (int i=0; i<ni; i++) d[i] = R[jj * cell_inci + (i + NGHOST)].rho;
            tdma_solve(a, b, c, d, x, ni);
            for (int i=0; i<ni; i++) R_temp[jj * cell_inci + (i + NGHOST)].rho = x[i];

            // Solve for rhou
            for (int i=0; i<ni; i++) d[i] = R[jj * cell_inci + (i + NGHOST)].rhou;
            tdma_solve(a, b, c, d, x, ni);
            for (int i=0; i<ni; i++) R_temp[jj * cell_inci + (i + NGHOST)].rhou = x[i];

            // Solve for rhov
            for (int i=0; i<ni; i++) d[i] = R[jj * cell_inci + (i + NGHOST)].rhov;
            tdma_solve(a, b, c, d, x, ni);
            for (int i=0; i<ni; i++) R_temp[jj * cell_inci + (i + NGHOST)].rhov = x[i];

            // Solve for rhoE
            for (int i=0; i<ni; i++) d[i] = R[jj * cell_inci + (i + NGHOST)].rhoE;
            tdma_solve(a, b, c, d, x, ni);
            for (int i=0; i<ni; i++) R_temp[jj * cell_inci + (i + NGHOST)].rhoE = x[i];
        }

        // --- j-sweep (Vertical) ---
        #pragma omp parallel for
        for (int ii = NGHOST; ii < ni + NGHOST; ii++) {
            // --- These vectors must be thread-private ---
            int max_dim = nj;
            std::vector<double> a(max_dim);
            std::vector<double> b(max_dim);
            std::vector<double> c(max_dim);
            std::vector<double> d(max_dim);
            std::vector<double> x(max_dim);
            // ------------------------------------------

            // Setup TDMA system for this i-line
            for (int j = 0; j < nj; j++) {
                a[j] = -epsilon;
                b[j] = 1.0 + 2.0 * epsilon;
                c[j] = -epsilon;
            }
            // Neumann BCs at ends (wall and farfield)
            a[0] = 0.0;
            c[nj-1] = 0.0;
            b[0] = 1.0 + epsilon;
            b[nj-1] = 1.0 + epsilon;

            // Solve for rho
            for (int j=0; j<nj; j++) d[j] = R_temp[(j + NGHOST) * cell_inci + ii].rho;
            tdma_solve(a, b, c, d, x, nj);
            for (int j=0; j<nj; j++) R[(j + NGHOST) * cell_inci + ii].rho = x[j];

            // Solve for rhou
            for (int j=0; j<nj; j++) d[j] = R_temp[(j + NGHOST) * cell_inci + ii].rhou;
            tdma_solve(a, b, c, d, x, nj);
            for (int j=0; j<nj; j++) R[(j + NGHOST) * cell_inci + ii].rhou = x[j];

            // Solve for rhov
            for (int j=0; j<nj; j++) d[j] = R_temp[(j + NGHOST) * cell_inci + ii].rhov;
            tdma_solve(a, b, c, d, x, nj);
            for (int j=0; j<nj; j++) R[(j + NGHOST) * cell_inci + ii].rhov = x[j];

            // Solve for rhoE
            for (int j=0; j<nj; j++) d[j] = R_temp[(j + NGHOST) * cell_inci + ii].rhoE;
            tdma_solve(a, b, c, d, x, nj);
            for (int j=0; j<nj; j++) R[(j + NGHOST) * cell_inci + ii].rhoE = x[j];
        }
    }
    
    ConservativeVars computeRoeFlux(const ConservativeVars& UL, const ConservativeVars& UR,
                                     double Sx, double Sy) const {
        PrimitiveVars WL = conservative2Primitive(UL);
        PrimitiveVars WR = conservative2Primitive(UR);
        
        double S_mag = std::sqrt(Sx*Sx + Sy*Sy) + EPS;
        double nx = Sx / S_mag;
        double ny = Sy / S_mag;
        
        // Roe average
        double RT = std::sqrt(UR.rho / UL.rho);
        double rho_roe = std::sqrt(UL.rho * UR.rho);
        double u_roe = (WL.u + RT * WR.u) / (1.0 + RT);
        double v_roe = (WL.v + RT * WR.v) / (1.0 + RT);
        double HL = (UL.rhoE + WL.p) / UL.rho;
        double HR = (UR.rhoE + WR.p) / UR.rho;
        double H_roe = (HL + RT * HR) / (1.0 + RT);
        double V2_roe = u_roe*u_roe + v_roe*v_roe;
        double a_roe = std::sqrt((GAMMA - 1.0) * (H_roe - 0.5 * V2_roe));
        
        if (a_roe < EPS || std::isnan(a_roe)) a_roe = a_inf;
        
        double Vn_roe = u_roe * nx + v_roe * ny;
        
        // Eigenvalues with entropy fix
        double lambda1 = std::abs(Vn_roe - a_roe);
        double lambda2 = std::abs(Vn_roe);
        double lambda3 = std::abs(Vn_roe + a_roe);
        
        double epsilon = 0.1 * a_roe;
        if (lambda1 < epsilon) lambda1 = 0.5 * (lambda1*lambda1/epsilon + epsilon);
        if (lambda2 < epsilon) lambda2 = 0.5 * (lambda2*lambda2/epsilon + epsilon);
        if (lambda3 < epsilon) lambda3 = 0.5 * (lambda3*lambda3/epsilon + epsilon);
        
        // Jump in variables
        double drho = UR.rho - UL.rho;
        double dp = WR.p - WL.p;
        double dvn = (WR.u - WL.u) * nx + (WR.v - WL.v) * ny;
        
        // Wave strengths
        double a2_roe = a_roe * a_roe;
        double alpha1 = 0.5 * (dp - rho_roe * a_roe * dvn) / a2_roe;
        double alpha2 = drho - dp / a2_roe;
        double alpha3 = 0.5 * (dp + rho_roe * a_roe * dvn) / a2_roe;
        
        // Physical flux average
        double rhoVn_L = WL.rho * (WL.u * nx + WL.v * ny);
        double rhoVn_R = WR.rho * (WR.u * nx + WR.v * ny);
        
        ConservativeVars F;
        F.rho = 0.5 * (rhoVn_L + rhoVn_R);
        F.rhou = 0.5 * (rhoVn_L * WL.u + WL.p * nx + rhoVn_R * WR.u + WR.p * nx);
        F.rhov = 0.5 * (rhoVn_L * WL.v + WL.p * ny + rhoVn_R * WR.v + WR.p * ny);
        F.rhoE = 0.5 * (rhoVn_L * HL + rhoVn_R * HR);
        
        // Roe dissipation
        double diss_rho = lambda1 * alpha1 + lambda2 * alpha2 + lambda3 * alpha3;
        double diss_u = lambda1 * alpha1 * (u_roe - a_roe * nx) + lambda2 * alpha2 * u_roe + 
                         lambda3 * alpha3 * (u_roe + a_roe * nx);
        double diss_v = lambda1 * alpha1 * (v_roe - a_roe * ny) + lambda2 * alpha2 * v_roe + 
                         lambda3 * alpha3 * (v_roe + a_roe * ny);
        double diss_E = lambda1 * alpha1 * (H_roe - a_roe * Vn_roe) + lambda2 * alpha2 * 0.5 * V2_roe +
                         lambda3 * alpha3 * (H_roe + a_roe * Vn_roe);
        
        // Final flux
        F.rho = (F.rho - 0.5 * diss_rho) * S_mag;
        F.rhou = (F.rhou - 0.5 * diss_u) * S_mag;
        F.rhov = (F.rhov - 0.5 * diss_v) * S_mag;
        F.rhoE = (F.rhoE - 0.5 * diss_E) * S_mag;
        
        return F;
    }
    
    bool checkSolution() {
        int cell_inci = mesh.cell_stride();
        
        bool is_valid = true;
        // This check is fast, rare, and needs to report coordinates,
        // so it's best to keep it serial.
        for (int jj = NGHOST; jj < mesh.njc + NGHOST; jj++) {
            for (int ii = NGHOST; ii < mesh.nic + NGHOST; ii++) {
                int idx = jj * cell_inci + ii;
                
                if (std::isnan(U[idx].rho) || std::isnan(U[idx].rhoE) ||
                    std::isinf(U[idx].rho) || std::isinf(U[idx].rhoE)) {
                    std::cerr << "NaN/Inf detected at (" << ii << "," << jj << ")" << std::endl;
                    is_valid = false;
                    break;
                }
                
                if (U[idx].rho < 0 || U[idx].rhoE < 0) {
                    std::cerr << "Negative density or energy at (" << ii << "," << jj << ")" << std::endl;
                    is_valid = false;
                    break;
                }
                
                PrimitiveVars W = conservative2Primitive(U[idx]);
                if (W.p < 0) {
                    std::cerr << "Negative pressure at (" << ii << "," << jj << "): p=" << W.p << std::endl;
                    is_valid = false;
                    break;
                }
            }
            if (!is_valid) break;
        }
        return is_valid;
    }
    
    
    // --- 4-Stage Runge-Kutta (Jameson) ---
    void explicitRK4() {
        int cell_inci = mesh.cell_stride();
        
        U_old = U;
        
        // Stage 1 (alpha = 1/4)
        applyBoundaryConditions();
        computeResiduals();
        smoothResiduals(); 
        
        #pragma omp parallel for
        for (int jj = NGHOST; jj < mesh.njc + NGHOST; jj++) {
            for (int ii = NGHOST; ii < mesh.nic + NGHOST; ii++) {
                int idx = jj * cell_inci + ii;
                double dt_vol = dt_local[idx] / mesh.vol[idx]; // Uses global dt
                U[idx] = U_old[idx] - (1.0/4.0) * dt_vol * R[idx];
            }
        }
        
        // Stage 2 (alpha = 1/3)
        applyBoundaryConditions();
        computeResiduals();
        smoothResiduals(); 
        
        #pragma omp parallel for
        for (int jj = NGHOST; jj < mesh.njc + NGHOST; jj++) {
            for (int ii = NGHOST; ii < mesh.nic + NGHOST; ii++) {
                int idx = jj * cell_inci + ii;
                double dt_vol = dt_local[idx] / mesh.vol[idx]; // Uses global dt
                U[idx] = U_old[idx] - (1.0/3.0) * dt_vol * R[idx];
            }
        }
        
        // Stage 3 (alpha = 1/2)
        applyBoundaryConditions();
        computeResiduals();
        smoothResiduals(); 
        
        #pragma omp parallel for
        for (int jj = NGHOST; jj < mesh.njc + NGHOST; jj++) {
            for (int ii = NGHOST; ii < mesh.nic + NGHOST; ii++) {
                int idx = jj * cell_inci + ii;
                double dt_vol = dt_local[idx] / mesh.vol[idx]; // Uses global dt
                U[idx] = U_old[idx] - (1.0/2.0) * dt_vol * R[idx];
            }
        }

        // Stage 4 (alpha = 1.0)
        applyBoundaryConditions();
        computeResiduals();
        smoothResiduals(); 
        
        #pragma omp parallel for
        for (int jj = NGHOST; jj < mesh.njc + NGHOST; jj++) {
            for (int ii = NGHOST; ii < mesh.nic + NGHOST; ii++) {
                int idx = jj * cell_inci + ii;
                double dt_vol = dt_local[idx] / mesh.vol[idx]; // Uses global dt
                U[idx] = U_old[idx] - (1.0) * dt_vol * R[idx];
            }
        }
    }
    
    // --- MODIFIED: L2 norm of continuity (rho) residual only ---
    double computeResidualNorm() {
        double L2_norm = 0.0;
        int cell_inci = mesh.cell_stride();
        int count = 0;
        
        // Using the smoothed residual for monitoring is acceptable.
        #pragma omp parallel for reduction(+:L2_norm, count)
        for (int jj = NGHOST; jj < mesh.njc + NGHOST; jj++) {
            for (int ii = NGHOST; ii < mesh.nic + NGHOST; ii++) {
                int idx = jj * cell_inci + ii;
                
                // Only sum the square of the continuity residual
                L2_norm += R[idx].rho * R[idx].rho;
                
                count++;
            }
        }
        
        // Normalize by the count (1.0 to ensure floating point division)
        return std::sqrt(L2_norm / (1.0 * count));
    }
    
    void solve() {
        std::cout << "\n=== Starting Solver ===" << std::endl;
        
        // Verify initial BCs
        applyBoundaryConditions();
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        double initial_residual = 0.0;
        
        for (int iter = 0; iter < max_iter; iter++) {
            computeTimeStep(); // <-- Computes global dt
            explicitRK4();     // <-- Call RK4
            applyBoundaryConditions();
            
            // Check for NaN/Inf
            if (!checkSolution()) {
                std::cerr << "Solution contains NaN or Inf at iteration " << iter << std::endl;
                break;
            }
            
            double residual = computeResidualNorm(); // Norm of (smoothed) residual
            
            if (iter == 0) {
                // For a proper initial residual, we need the unsmoothed one
                computeResiduals();
                initial_residual = computeResidualNorm(); // This will now use the new norm
            }
            if (initial_residual < EPS) initial_residual = 1.0;
            
            residual_history.push_back(residual);
            
            if (iter % 100 == 0) {
                std::cout << "Iter: " << std::setw(6) << iter 
                          << " | Residual: " << std::scientific << std::setprecision(6) << residual  << std::endl;
            }
            
            
            
            if (residual < conv_tol || residual / initial_residual < 1e-10) {
                std::cout << "\nConverged at iteration " << iter << std::endl;
                break;
            }
            
            if (residual > 1e10) {
                std::cerr << "Solution diverged at iteration " << iter << std::endl;
                break;
            }
        }
        
        // Final verification
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "\nExecution time: " << duration.count() / 1000.0 << " seconds" << std::endl;
        std::cout << "Total iterations: " << residual_history.size() << std::endl;
    }
    
    // --- MODIFIED: Added CL0_PG for Prandtl-Glauert correction output ---
    void computeForces(double& CL, double& CD, double& CM, double& CL0_PG) {
        CL = CD = CM = CL0_PG = 0.0;
        int cell_inci = mesh.cell_stride();
        int node_inci = mesh.node_stride();
        double chord = 1.0;
        double x_ref = 0.25 * chord;

        double Fx = 0.0, Fy = 0.0, M_ref = 0.0;

        // integrate pressure on airfoil surface (j = NGHOST face line)
        #pragma omp parallel for reduction(+:Fx, Fy, M_ref)
        for (int ii = NGHOST; ii < mesh.nic + NGHOST; ii++) {
            int cell_above = NGHOST * cell_inci + ii;      // interior cell above the j-min face
            int cell_below = (NGHOST - 1) * cell_inci + ii; // cell below the face (ghost or interior)

            // primitive vars for both sides -> average for face
            PrimitiveVars W_above = conservative2Primitive(U[cell_above]);
            PrimitiveVars W_below = conservative2Primitive(U[cell_below]);

            double p_face = 0.5 * (W_above.p + W_below.p);

            // face normal (Sx_j, Sy_j) stored for that face index
            int face_idx = NGHOST * cell_inci + ii;
            double Sx = mesh.Sx_j[face_idx];
            double Sy = mesh.Sy_j[face_idx];

            // pressure force on body: -p * S_vec
            double Fx_p = -p_face * Sx;
            double Fy_p = -p_face * Sy;

            Fx += Fx_p;
            Fy += Fy_p;

            // face center from nodes (consistent with face_idx in j-direction)
            int n1 = NGHOST * node_inci + ii;
            int n2 = NGHOST * node_inci + (ii + 1);
            double x_face_center = 0.5 * (mesh.x[n1] + mesh.x[n2]);
            double y_face_center = 0.5 * (mesh.y[n1] + mesh.y[n2]);

            // moment about (x_ref, 0): r x F
            double rx = x_face_center - x_ref;
            double ry = y_face_center; // y_ref = 0
            M_ref += rx * Fy_p - ry * Fx_p;
        }

        // dynamic pressure (explicit and unambiguous)
        double Vinf2 = u_inf * u_inf + v_inf * v_inf;
        double q_inf = 0.5 * rho_inf * Vinf2;
        double S_ref = chord * 1.0;

        // transform to wind axes
        double cosa = std::cos(alpha);
        double sina = std::sin(alpha);

        double F_lift = -Fx * sina + Fy * cosa;
        double F_drag =  Fx * cosa + Fy * sina;

        CL = F_lift / (q_inf * S_ref);
        CD = F_drag / (q_inf * S_ref);
        CM = M_ref / (q_inf * S_ref * chord);
        
        // --- NEW: Prandtl-Glauert Correction ---
        if (M_inf < 1.0 && M_inf > EPS) {
            // C_L,0 = C_L / (1/beta) = C_L * beta
            CL0_PG = CL * beta_PG;
        } else {
            CL0_PG = CL; // Equivalent to incompressible or M=0
        }

        std::cout << "\n=== Force Coefficients ===" << std::endl;
        std::cout << "M_inf: " << std::fixed << std::setprecision(4) << M_inf << ", Beta_PG: " << beta_PG << std::endl;
        std::cout << "CL (Compressible) = " << std::fixed << std::setprecision(8) << CL << std::endl;
        std::cout << "CD (Compressible) = " << CD << std::endl;
        std::cout << "CM (Compressible) = " << CM << std::endl;
        std::cout << "CL,0 (P-G Corrected Incomp. Equivalent) = " << CL0_PG << std::endl;
        // --- END NEW ---
    }
    
    void writeTecplotSolution(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open output file " << filename << std::endl;
            return;
        }

        file << "TITLE = \"Euler Solution (Node-Centered): M=" << M_inf << ", alpha=" << alpha*180/PI << " deg\"\n";
        file << "VARIABLES = \"x\", \"y\", \"rho\", \"u\", \"v\", \"p\", \"Mach\", \"Cp\"\n";
        file << "ZONE I=" << mesh.ni << ", J=" << mesh.nj << ", F=BLOCK\n";

        int cell_inci = mesh.cell_stride();
        int node_inci = mesh.node_stride();

        // --- FILE I/O MUST BE SERIAL ---

        // x-coordinates (Node data)
        for (int jj = NGHOST; jj < mesh.nj + NGHOST; jj++) {
            for (int ii = NGHOST; ii < mesh.ni + NGHOST; ii++) {
                int index = jj * node_inci + ii;
                file << std::setprecision(10) << mesh.x[index] << " ";
            }
            file << "\n";
        }

        // y-coordinates (Node data)
        for (int jj = NGHOST; jj < mesh.nj + NGHOST; jj++) {
            for (int ii = NGHOST; ii < mesh.ni + NGHOST; ii++) {
                int index = jj * node_inci + ii;
                file << std::setprecision(10) << mesh.y[index] << " ";
            }
            file << "\n";
        }

        // Pre-calculate primitive vars for all cells (including ghosts)
        struct CellPrimitives { double rho, u, v, p, mach, cp; };
        std::vector<CellPrimitives> cell_data(U.size());

        #pragma omp parallel for
        for (size_t i = 0; i < U.size(); ++i) {
            PrimitiveVars W = conservative2Primitive(U[i]);
            double a = std::sqrt(GAMMA * W.p / W.rho);
            double M_local = std::sqrt(W.u*W.u + W.v*W.v) / (a + EPS);
            double Cp = (W.p - p_inf) / (0.5 * GAMMA * p_inf * M_inf * M_inf);
            cell_data[i] = {W.rho, W.u, W.v, W.p, M_local, Cp};
        }

        // Helper function to get the 4 cell indices surrounding a node (jj, ii)
        auto get_cell_indices = [&](int jj, int ii) {
            int cell1 = (jj - 1) * cell_inci + (ii - 1); // Bottom-left cell
            int cell2 = (jj) * cell_inci + (ii - 1);     // Top-left cell
            int cell3 = (jj - 1) * cell_inci + (ii);     // Bottom-right cell
            int cell4 = (jj) * cell_inci + (ii);         // Top-right cell
            return std::make_tuple(cell1, cell2, cell3, cell4);
        };

        // --- FILE I/O MUST BE SERIAL ---
        
        // rho (Interpolated)
        for (int jj = NGHOST; jj < mesh.nj + NGHOST; jj++) {
            for (int ii = NGHOST; ii < mesh.ni + NGHOST; ii++) {
                auto [c1, c2, c3, c4] = get_cell_indices(jj, ii);
                double rho_avg = 0.25 * (
                    cell_data[c1].rho + cell_data[c2].rho +
                    cell_data[c3].rho + cell_data[c4].rho
                );
                file << std::setprecision(10) << rho_avg << " ";
            }
            file << "\n";
        }

        // u (Interpolated)
        for (int jj = NGHOST; jj < mesh.nj + NGHOST; jj++) {
            for (int ii = NGHOST; ii < mesh.ni + NGHOST; ii++) {
                auto [c1, c2, c3, c4] = get_cell_indices(jj, ii);
                double u_avg = 0.25 * (
                    cell_data[c1].u + cell_data[c2].u +
                    cell_data[c3].u + cell_data[c4].u
                );
                file << std::setprecision(10) << u_avg << " ";
            }
            file << "\n";
        }

        // v (Interpolated)
        for (int jj = NGHOST; jj < mesh.nj + NGHOST; jj++) {
            for (int ii = NGHOST; ii < mesh.ni + NGHOST; ii++) {
                auto [c1, c2, c3, c4] = get_cell_indices(jj, ii);
                double v_avg = 0.25 * (
                    cell_data[c1].v + cell_data[c2].v +
                    cell_data[c3].v + cell_data[c4].v
                );
                file << std::setprecision(10) << v_avg << " ";
            }
            file << "\n";
        }

        // p (Interpolated)
        for (int jj = NGHOST; jj < mesh.nj + NGHOST; jj++) {
            for (int ii = NGHOST; ii < mesh.ni + NGHOST; ii++) {
                auto [c1, c2, c3, c4] = get_cell_indices(jj, ii);
                double p_avg = 0.25 * (
                    cell_data[c1].p + cell_data[c2].p +
                    cell_data[c3].p + cell_data[c4].p
                );
                file << std::setprecision(10) << p_avg << " ";
            }
            file << "\n";
        }

        // Mach number (Interpolated)
        for (int jj = NGHOST; jj < mesh.nj + NGHOST; jj++) {
            for (int ii = NGHOST; ii < mesh.ni + NGHOST; ii++) {
                auto [c1, c2, c3, c4] = get_cell_indices(jj, ii);
                double m_avg = 0.25 * (
                    cell_data[c1].mach + cell_data[c2].mach +
                    cell_data[c3].mach + cell_data[c4].mach
                );
                file << std::setprecision(10) << m_avg << " ";
            }
            file << "\n";
        }

        // Cp (Interpolated)
        for (int jj = NGHOST; jj < mesh.nj + NGHOST; jj++) {
            for (int ii = NGHOST; ii < mesh.ni + NGHOST; ii++) {
                auto [c1, c2, c3, c4] = get_cell_indices(jj, ii);
                double cp_avg = 0.25 * (
                    cell_data[c1].cp + cell_data[c2].cp +
                    cell_data[c3].cp + cell_data[c4].cp
                );
                file << std::setprecision(10) << cp_avg << " ";
            }
            file << "\n";
        }

        file.close();
        std::cout << "Node-centered solution written to " << filename << std::endl;
    }
    
    void writeSurfacePressure(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open output file " << filename << std::endl;
            return;
        }
        
        file << "# x/c, Cp (NACA0012 surface)\n";
        
        int cell_inci = mesh.cell_stride();
        int jj_wall = NGHOST;
        
        std::vector<std::pair<double, double>> cp_data;
        
        for (int ii = NGHOST; ii < mesh.nic + NGHOST; ii++) {
            int idx = jj_wall * cell_inci + ii;
            PrimitiveVars W = conservative2Primitive(U[idx]);
            
            double x = mesh.xc[idx];
            double Cp = (W.p - p_inf) / (0.5 * GAMMA * p_inf * M_inf * M_inf);
            
            cp_data.push_back({x, Cp});
        }
        
        // Sort by x-coordinate
        std::sort(cp_data.begin(), cp_data.end());
        
        for (const auto& pt : cp_data) {
            file << std::setprecision(10) << pt.first << " " << pt.second << "\n";
        }
        
        file.close();
        std::cout << "Surface pressure written to " << filename << std::endl;
    }
    
    void writeResidualHistory(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) return;
        
        file << "# Iteration, Residual\n";
        for (size_t i = 0; i < residual_history.size(); i++) {
            file << i << " " << std::scientific << std::setprecision(10) 
                 << residual_history[i] << "\n";
        }
        
        file.close();
        std::cout << "Residual history written to " << filename << std::endl;
    }
};

int main(int argc, char** argv) {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << " 2D Euler Solver - Cell-Based O-Grid \n";
    std::cout << " (GTS + Res. Smoothing + OpenMP) \n";
    std::cout << " (RK4 Scheme w/ Prandtl-Glauert Output) \n";
    std::cout << "========================================\n\n";
    
    std::string mesh_file = "NACA2412.x";
    double Mach = 0.2;
    double alpha = 16.5;
    double CFL = 3.0;
    
    
    // Parse command line arguments
    if (argc > 1) mesh_file = argv[1];
    if (argc > 2) Mach = std::atof(argv[2]);
    if (argc > 3) alpha = std::atof(argv[3]);
    if (argc > 4) CFL = std::atof(argv[4]);
    
    std::cout << "Input Parameters:\n";
    std::cout << " Mesh file: " << mesh_file << "\n";
    std::cout << " Mach number: " << Mach << "\n";
    std::cout << " Angle of attack: " << alpha << " deg\n";
    std::cout << " CFL number: " << CFL << "\n\n";
    
    try {
        // Create solver
        EulerSolver solver(mesh_file, Mach, alpha, CFL);
        
        // Solve
        solver.solve();
        
        // Compute forces
        double CL, CD, CM, CL0_PG;
        // NOTE: The computeForces signature has been updated to include CL0_PG
        solver.computeForces(CL, CD, CM, CL0_PG);
        
        // Write output
        solver.writeTecplotSolution("solution.dat");
        solver.writeSurfacePressure("surface_cp.dat");
        solver.writeResidualHistory("residual.dat");
        
        std::cout << "\n=== Simulation Complete ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}