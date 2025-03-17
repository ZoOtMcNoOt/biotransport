#ifndef BIOTRANSPORT_MESH_HPP
#define BIOTRANSPORT_MESH_HPP

#include <vector>
#include <stdexcept>

namespace biotransport {

/**
 * A structured mesh for 1D and 2D simulations.
 */
class StructuredMesh {
public:
    /**
     * Create a 1D structured mesh.
     * 
     * @param nx Number of cells in x direction
     * @param xmin Minimum x coordinate
     * @param xmax Maximum x coordinate
     */
    StructuredMesh(int nx, double xmin, double xmax);
    
    /**
     * Create a 2D structured mesh.
     * 
     * @param nx Number of cells in x direction
     * @param ny Number of cells in y direction
     * @param xmin Minimum x coordinate
     * @param xmax Maximum x coordinate
     * @param ymin Minimum y coordinate
     * @param ymax Maximum y coordinate
     */
    StructuredMesh(int nx, int ny, double xmin, double xmax, 
                  double ymin, double ymax);
    
    /**
     * Get the number of nodes in the mesh.
     */
    int numNodes() const;
    
    /**
     * Get the number of cells in the mesh.
     */
    int numCells() const;
    
    /**
     * Get the cell size in x direction.
     */
    double dx() const { return dx_; }
    
    /**
     * Get the cell size in y direction.
     */
    double dy() const { return dy_; }
    
    /**
     * Check if this is a 1D mesh.
     */
    bool is1D() const { return is_1d_; }
    
    /**
     * Get the x coordinate of a node.
     */
    double x(int i) const;
    
    /**
     * Get the y coordinate of a node.
     */
    double y(int i, int j) const;
    
    /**
     * Get the global index of a node.
     */
    int index(int i, int j = 0) const;
    
    /**
     * Get the dimensions of the mesh.
     */
    int nx() const { return nx_; }
    int ny() const { return ny_; }
    
private:
    int nx_, ny_;              // Number of cells in each direction
    double xmin_, xmax_;       // x coordinate range
    double ymin_, ymax_;       // y coordinate range
    double dx_, dy_;           // Cell sizes
    bool is_1d_;               // Is this a 1D mesh?
};

} // namespace biotransport

#endif // BIOTRANSPORT_MESH_HPP