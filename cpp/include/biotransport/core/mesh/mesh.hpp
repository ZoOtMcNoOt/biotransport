#ifndef BIOTRANSPORT_CORE_MESH_MESH_HPP
#define BIOTRANSPORT_CORE_MESH_MESH_HPP

// Base mesh interface/class
// In the target architecture, this would be a base class that StructuredMesh inherits from
// For now, we'll use it as a bridge to structured_mesh.hpp

#include <biotransport/core/mesh/structured_mesh.hpp>

namespace biotransport {
    // Type alias for backward compatibility
    using Mesh = StructuredMesh;
}

#endif // BIOTRANSPORT_CORE_MESH_MESH_HPP