#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <limits>
#include <omp.h> // <-- NEW: Include OpenMP header

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

    // --- NEW: Operator overloads (needed for RK4 math) ---
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
// --- End of new operators ---


// Struct to hold primitive variables
struct PrimitiveVars {
    double rho;
    double u;
    double v;
    double p;
    
    PrimitiveVars() : rho(0), u(0), v(0), p(0) {}
};


struct Mesh {
    int ni, nj;      // nodes in i and j directions
    int nic, njc;    // cells in i and j directions
    
    std::vector<double> x, y;      // node coordinates [node_stride = ni+2*NGHOST]
    std::vector<double> xc, yc;    // cell center coordinates [cell_stride = nic+2*NGHOST]
    std::vector<double> vol;       // cell volumes
    
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
    double rho_inf, u_inf, v_inf, p_inf, a_inf, E_inf, T_inf;
    
    std::vector<ConservativeVars> U;      // Conservative variables (cell-centered)
    std::vector<ConservativeVars> U_old;  // Old time level
    std::vector<ConservativeVars> R;      // Residuals
    std::vector<ConservativeVars> R_temp; // Buffer for residual smoothing
    std::vector<double> dt_local;         // Local time step (set to global)
    
    double CFL;
    int max_iter;
    double conv_tol;
    
    // Residual Smoothing Parameters
    bool use_residual_smoothing;
    double res_smoothing_coeff;
    
    // --- JST Members ---
    bool use_artificial_dissipation;
    double k2, k4; // 2nd and 4th order coefficients
    std::vector<double> spectral_radius_cell; // Cell-centered spectral radius
    std::vector<double> pressure_sensor;      // Cell-centered pressure sensor
    std::vector<ConservativeVars> D_i;        // Dissipation flux at i-faces
    std::vector<ConservativeVars> D_j;        // Dissipation flux at j-faces
    // -------------------

    std::vector<double> residual_history;
    
public:
    EulerSolver(const std::string& mesh_file, double Mach, double alpha_deg, double CFL_num = 0.5)
        : M_inf(Mach), alpha(alpha_deg * PI / 180.0), CFL(CFL_num), 
          max_iter(50000), conv_tol(1e-7),
          use_residual_smoothing(true), res_smoothing_coeff(2.0),
          use_artificial_dissipation(true), k2(0.25), k4(0.01) { // <-- Init JST
        
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
        R_temp.resize(ncells); 
        dt_local.resize(ncells, 0.0);
        
        // --- NEW: Resize JST buffers ---
        spectral_radius_cell.resize(ncells, 0.0);
        pressure_sensor.resize(ncells, 0.0);
        D_i.resize(ncells); // i-face flux
        D_j.resize(ncells); // j-face flux
        // ---------------------------------
        
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
    
    // --- Global Time Stepping (Also computes spectral radius) ---
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
                
                spectral_radius_cell[idx] = V_mag + a;
                
                double lambda_max = spectral_radius_cell[idx];
                double h = std::sqrt(mesh.vol[idx]);
                
                double dt_val = CFL * h / (lambda_max + EPS);
                dt_local[idx] = dt_val;
                
                if (dt_val < dt_min) {
                    dt_min = dt_val;
                }
            }
        }
        
        // --- Global Time Step ---
        #pragma omp parallel for
        for (int jj = NGHOST; jj < mesh.njc + NGHOST; jj++) {
            for (int ii = NGHOST; ii < mesh.nic + NGHOST; ii++) {
                dt_local[jj * cell_inci + ii] = dt_min;
            }
        }
    }
    
    ConservativeVars computeConvectiveFlux(const PrimitiveVars& W, double Sx, double Sy) const {
        ConservativeVars F_conv;
        
        // Project velocity onto face normal
        double Vn = W.u * Sx + W.v * Sy;
        
        // Compute Euler flux F*Sx + G*Sy
        F_conv.rho = Vn * W.rho;
        F_conv.rhou = Vn * W.rho * W.u + W.p * Sx;
        F_conv.rhov = Vn * W.rho * W.v + W.p * Sy;
        
        // H = E + p/rho
        double E = W.p / ((GAMMA - 1.0) * W.rho) + 0.5 * (W.u*W.u + W.v*W.v);
        double H = E + W.p / W.rho;
        F_conv.rhoE = Vn * W.rho * H;
        
        return F_conv;
    }

    
    void computeResiduals() {
        int cell_inci = mesh.cell_stride();
        
        // Reset residuals
        #pragma omp parallel for
        for (size_t i = 0; i < R.size(); i++) {
            R[i].rho = R[i].rhou = R[i].rhov = R[i].rhoE = 0.0;
        }
        
        // i-direction fluxes (Central Difference)
        #pragma omp parallel for
        for (int jj = NGHOST; jj < mesh.njc + NGHOST; jj++) {
            for (int ii = NGHOST; ii <= mesh.nic + NGHOST; ii++) {
                int idx_L = jj * cell_inci + (ii-1);
                int idx_R = jj * cell_inci + ii;
                int face_idx = jj * cell_inci + ii;
                
                double Sx = mesh.Sx_i[face_idx];
                double Sy = mesh.Sy_i[face_idx];
                
                PrimitiveVars W_L = conservative2Primitive(U[idx_L]);
                PrimitiveVars W_R = conservative2Primitive(U[idx_R]);
                
                auto flux_L = computeConvectiveFlux(W_L, Sx, Sy);
                auto flux_R = computeConvectiveFlux(W_R, Sx, Sy);
                
                // Central difference flux
                auto flux = 0.5 * (flux_L + flux_R);
                
                if (ii-1 >= NGHOST && ii-1 < mesh.nic + NGHOST) {
                    // Thread-safe because jj is the outer parallel loop
                    R[idx_L] += flux;
                }
                
                if (ii >= NGHOST && ii < mesh.nic + NGHOST) {
                    R[idx_R] -= flux;
                }
            }
        }
        
        // j-direction fluxes (Central Difference)
        // --- LOOP ORDER SWAPPED for parallel safety ---
        #pragma omp parallel for
        for (int ii = NGHOST; ii < mesh.nic + NGHOST; ii++) {
            for (int jj = NGHOST; jj <= mesh.njc + NGHOST; jj++) {
                int idx_L = (jj-1) * cell_inci + ii; // "Bottom" cell
                int idx_R = jj * cell_inci + ii;     // "Top" cell
                int face_idx = jj * cell_inci + ii;
                
                double Sx = mesh.Sx_j[face_idx];
                double Sy = mesh.Sy_j[face_idx];
                
                PrimitiveVars W_L = conservative2Primitive(U[idx_L]);
                PrimitiveVars W_R = conservative2Primitive(U[idx_R]);
                
                auto flux_L = computeConvectiveFlux(W_L, Sx, Sy);
                auto flux_R = computeConvectiveFlux(W_R, Sx, Sy);
                
                // Central difference flux
                auto flux = 0.5 * (flux_L + flux_R);
                
                if (jj-1 >= NGHOST && jj-1 < mesh.njc + NGHOST) {
                    // Thread-safe because ii is the outer parallel loop
                    R[idx_L] += flux;
                }
                
                if (jj >= NGHOST && jj < mesh.njc + NGHOST) {
                    R[idx_R] -= flux;
                }
            }
        }
    }

    
    void addArtificialDissipation() {
        if (!use_artificial_dissipation) return;
        
        int cell_inci = mesh.cell_stride();

        // --- Part 1: Compute i-direction pressure sensor (cell-centered) ---
        // (Uses p at i-1, i, i+1)
        #pragma omp parallel for
        for (int jj = NGHOST; jj < mesh.njc + NGHOST; jj++) {
            for (int ii = NGHOST; ii < mesh.nic + NGHOST; ii++) {
                int idx = jj * cell_inci + ii;
                int idx_im1 = jj * cell_inci + (ii - 1);
                int idx_ip1 = jj * cell_inci + (ii + 1);
                
                double p_im1 = conservative2Primitive(U[idx_im1]).p;
                double p_i   = conservative2Primitive(U[idx]).p;
                double p_ip1 = conservative2Primitive(U[idx_ip1]).p;
                
                double num = std::abs(p_ip1 - 2.0 * p_i + p_im1);
                double den = p_ip1 + 2.0 * p_i + p_im1 + EPS;
                pressure_sensor[idx] = num / den;
            }
        }
        
        // --- Part 2: Compute i-dissipation flux D_i (at i+1/2 faces) ---
        // (Uses U at i-1, i, i+1, i+2)
        #pragma omp parallel for
        for (int jj = NGHOST; jj < mesh.njc + NGHOST; jj++) {
            for (int ii = NGHOST - 1; ii < mesh.nic + NGHOST; ii++) {
                int idx_face = jj * cell_inci + ii; // This is face i+1/2
                
                int idx_im1 = jj * cell_inci + (ii - 1);
                int idx_i   = jj * cell_inci + (ii);
                int idx_ip1 = jj * cell_inci + (ii + 1);
                int idx_ip2 = jj * cell_inci + (ii + 2);
                
                // Sensor at i and i+1
                double nu_i   = pressure_sensor[idx_i];
                double nu_ip1 = pressure_sensor[idx_ip1];
                double nu_max = std::max(nu_i, nu_ip1);
                
                double eps2 = k2 * nu_max;
                double eps4 = std::max(0.0, k4 - eps2);
                
                double lambda = 0.5 * (spectral_radius_cell[idx_i] + spectral_radius_cell[idx_ip1]);
                
                ConservativeVars diff1 = U[idx_ip1] - U[idx_i];
                ConservativeVars diff3 = (U[idx_ip2] - 3.0*U[idx_ip1]) + (3.0*U[idx_i] - U[idx_im1]);
                
                D_i[idx_face] = 0.5 * lambda * (eps2 * diff1 - eps4 * diff3);
            }
        }
        
        // --- Part 3: Compute j-direction pressure sensor (cell-centered) ---
        // (Uses p at j-1, j, j+1)
        #pragma omp parallel for
        for (int jj = NGHOST; jj < mesh.njc + NGHOST; jj++) {
            for (int ii = NGHOST; ii < mesh.nic + NGHOST; ii++) {
                int idx = jj * cell_inci + ii;
                int idx_jm1 = (jj - 1) * cell_inci + ii;
                int idx_jp1 = (jj + 1) * cell_inci + ii;
                
                double p_jm1 = conservative2Primitive(U[idx_jm1]).p;
                double p_j   = conservative2Primitive(U[idx]).p;
                double p_jp1 = conservative2Primitive(U[idx_jp1]).p;
                
                double num = std::abs(p_jp1 - 2.0 * p_j + p_jm1);
                double den = p_jp1 + 2.0 * p_j + p_jm1 + EPS;
                
                // Combine with i-direction sensor
                pressure_sensor[idx] = std::max(pressure_sensor[idx], num / den);
            }
        }
        
        // --- Part 4: Compute j-dissipation flux D_j (at j+1/2 faces) ---
        // (Uses U at j-1, j, j+1, j+2)
        #pragma omp parallel for
        for (int jj = NGHOST - 1; jj < mesh.njc + NGHOST; jj++) {
            for (int ii = NGHOST; ii < mesh.nic + NGHOST; ii++) {
                int idx_face = jj * cell_inci + ii; // This is face j+1/2
                
                int idx_jm1 = (jj - 1) * cell_inci + ii;
                int idx_j   = (jj)     * cell_inci + ii;
                int idx_jp1 = (jj + 1) * cell_inci + ii;
                int idx_jp2 = (jj + 2) * cell_inci + ii;
                
                double nu_j   = pressure_sensor[idx_j];
                double nu_jp1 = pressure_sensor[idx_jp1];
                double nu_max = std::max(nu_j, nu_jp1);
                
                double eps2 = k2 * nu_max;
                double eps4 = std::max(0.0, k4 - eps2);
                
                double lambda = 0.5 * (spectral_radius_cell[idx_j] + spectral_radius_cell[idx_jp1]);
                
                ConservativeVars diff1 = U[idx_jp1] - U[idx_j];
                ConservativeVars diff3 = (U[idx_jp2] - 3.0*U[idx_jp1]) + (3.0*U[idx_j] - U[idx_jm1]);
                
                D_j[idx_face] = 0.5 * lambda * (eps2 * diff1 - eps4 * diff3);
            }
        }
        
        // --- Part 5: Add dissipation to residual R ---
        #pragma omp parallel for
        for (int jj = NGHOST; jj < mesh.njc + NGHOST; jj++) {
            for (int ii = NGHOST; ii < mesh.nic + NGHOST; ii++) {
                int idx = jj * cell_inci + ii;
                int idx_im1 = jj * cell_inci + (ii - 1); // i-1/2 face
                int idx_jm1 = (jj - 1) * cell_inci + ii; // j-1/2 face
                
                R[idx] -= (D_i[idx] - D_i[idx_im1]);
                R[idx] -= (D_j[idx] - D_j[idx_jm1]);
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
        
        // --- Stage 1 (alpha = 1/4) ---
        applyBoundaryConditions();
        computeResiduals();         // Convective flux
        addArtificialDissipation(); // Dissipative flux
        smoothResiduals();          // Residual smoothing
        
        #pragma omp parallel for
        for (int jj = NGHOST; jj < mesh.njc + NGHOST; jj++) {
            for (int ii = NGHOST; ii < mesh.nic + NGHOST; ii++) {
                int idx = jj * cell_inci + ii;
                double dt_vol = dt_local[idx] / mesh.vol[idx]; // Uses global dt
                U[idx] = U_old[idx] - (1.0/4.0) * dt_vol * R[idx];
            }
        }
        
        // --- Stage 2 (alpha = 1/3) ---
        applyBoundaryConditions();
        computeResiduals();
        addArtificialDissipation();
        smoothResiduals(); 
        
        #pragma omp parallel for
        for (int jj = NGHOST; jj < mesh.njc + NGHOST; jj++) {
            for (int ii = NGHOST; ii < mesh.nic + NGHOST; ii++) {
                int idx = jj * cell_inci + ii;
                double dt_vol = dt_local[idx] / mesh.vol[idx]; // Uses global dt
                U[idx] = U_old[idx] - (1.0/3.0) * dt_vol * R[idx];
            }
        }
        
        // --- Stage 3 (alpha = 1/2) ---
        applyBoundaryConditions();
        computeResiduals();
        addArtificialDissipation();
        smoothResiduals(); 
        
        #pragma omp parallel for
        for (int jj = NGHOST; jj < mesh.njc + NGHOST; jj++) {
            for (int ii = NGHOST; ii < mesh.nic + NGHOST; ii++) {
                int idx = jj * cell_inci + ii;
                double dt_vol = dt_local[idx] / mesh.vol[idx]; // Uses global dt
                U[idx] = U_old[idx] - (1.0/2.0) * dt_vol * R[idx];
            }
        }

        // --- Stage 4 (alpha = 1.0) ---
        applyBoundaryConditions();
        computeResiduals();
        addArtificialDissipation();
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
            computeTimeStep(); 
            explicitRK4();     
            applyBoundaryConditions();
            
            if (!checkSolution()) {
                std::cerr << "Solution contains NaN or Inf at iteration " << iter << std::endl;
                break;
            }
            
            double residual = computeResidualNorm(); 
            
            if (iter == 0) {
                computeResiduals();
                addArtificialDissipation(); // Must add AD to get correct initial R
                initial_residual = computeResidualNorm();
            }
            if (initial_residual < EPS) initial_residual = 1.0;
            
            residual_history.push_back(residual);
            
            if (iter % 100 == 0) {
                std::cout << "Iter: " << std::setw(6) << iter 
                         << " | Residual: " << std::scientific << std::setprecision(6) << residual
                         << " | Reduction: " << std::fixed << std::setprecision(3) 
                         << residual / initial_residual << std::endl;
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
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "\nExecution time: " << duration.count() / 1000.0 << " seconds" << std::endl;
        std::cout << "Total iterations: " << residual_history.size() << std::endl;
    }
    
    void computeForces(double& CL, double& CD, double& CM) {
        CL = CD = CM = 0.0;
        int cell_inci = mesh.cell_stride();
        int node_inci = mesh.node_stride();
        double chord = 1.0;
        double x_ref = 0.25 * chord;

        double Fx = 0.0, Fy = 0.0, M_ref = 0.0;

        // integrate pressure on airfoil surface (j = NGHOST face line)
        #pragma omp parallel for reduction(+:Fx, Fy, M_ref)
        for (int ii = NGHOST; ii < mesh.nic + NGHOST; ii++) {
            int cell_above = NGHOST * cell_inci + ii;       // interior cell above the j-min face
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

        std::cout << "\n=== Force Coefficients ===" << std::endl;
        std::cout << "CL = " << std::fixed << std::setprecision(6) << CL << std::endl;
        std::cout << "CD = " << CD << std::endl;
        std::cout << "CM = " << CM << std::endl;
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
    std::cout << "  2D Euler Solver - Cell-Based O-Grid  \n";
    std::cout << "  (JST Central Scheme + IRS + GTS)     \n";
    std::cout << "  (RK4 Scheme + OpenMP)                \n";
    std::cout << "========================================\n\n";
    
    // Default parameters
    std::string mesh_file = "33x33.x";
    double Mach = 0.5;
    double alpha = 1.25;
    double CFL = 3.0; // Using stable 3.0
    
    // Parse command line arguments
    if (argc > 1) mesh_file = argv[1];
    if (argc > 2) Mach = std::atof(argv[2]);
    if (argc > 3) alpha = std::atof(argv[3]);
    if (argc > 4) CFL = std::atof(argv[4]);
    
    std::cout << "Input Parameters:\n";
    std::cout << "  Mesh file: " << mesh_file << "\n";
    std::cout << "  Mach number: " << Mach << "\n";
    std::cout << "  Angle of attack: " << alpha << " deg\n";
    std::cout << "  CFL number: " << CFL << "\n\n";
    
    try {
        // Create solver
        EulerSolver solver(mesh_file, Mach, alpha, CFL);
        
        // Solve
        solver.solve();
        
        // Compute forces
        double CL, CD, CM;
        solver.computeForces(CL, CD, CM);
        
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