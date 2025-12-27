# BMEN 341 Biotransport Course Analysis for BioTransport Library

## Document Purpose

This document provides a comprehensive analysis of Texas A&M University's BMEN 341 (Biotransport) course materials to inform the development and enhancement of the BioTransport computational library. The analysis covers:

1. **Complete topic coverage** from the course syllabus
2. **Detailed problem decomposition** from all homework assignments (HW1-HW6)
3. **Practice problem analysis** (NASA Bioreactor, Poiseuille Flow)
4. **Equation sheet analysis** for core formulations
5. **Implementation recommendations** mapping each concept to library features

---

## Table of Contents

1. [Course Overview and Structure](#1-course-overview-and-structure)
2. [Part I: Mass Transport Fundamentals](#2-part-i-mass-transport-fundamentals)
3. [Part II: Fluid Mechanics and Biofluids](#3-part-ii-fluid-mechanics-and-biofluids)
4. [Homework Problem Analysis](#4-homework-problem-analysis)
5. [Practice Problem Deep Dives](#5-practice-problem-deep-dives)
6. [Core Equations Reference](#6-core-equations-reference)
7. [Implementation Gap Analysis](#7-implementation-gap-analysis)
8. [Library Extensions (Status)](#8-library-extensions-status)

---

## 1. Course Overview and Structure

### 1.1 Course Description

BMEN 341 covers the fundamentals of momentum, mass, and energy transport in living and biomedical systems. The course examines:

- **Length scales**: Intracellular to organ level
- **Core topics**: 
  - Fluid mechanics
  - Transport by diffusion
  - Effects of convection, electrochemical potential, and chemical reactions
  - Energy-tissue interactions

### 1.2 Course Learning Outcomes

Upon completion, students should be able to:

1. Mathematically define and describe general biotransport problems
2. Derive governing equations with appropriate boundary/initial conditions
3. Solve and analyze a variety of biotransport problems
4. Develop transport models for biomedical applications
5. Explain fluid mechanics in biomechanics (kinematics, Cauchy stress, momentum balance, constitutive relations, boundary conditions)

### 1.3 Course Schedule and Topic Breakdown

| Week | Topics | Relevance to Library |
|------|--------|---------------------|
| 1 | Mathematical preliminaries, Diffusion vs. convection | Core math utilities, solver foundations |
| 2 | Balance of mass (Continuity), Fick's law | `DiffusionSolver` foundation |
| 3 | Steady-state diffusion 1D, Partition coefficient | Membrane transport |
| 4 | Unsteady diffusion, Biot/Fourier modulus | Time-stepping algorithms |
| 5 | Convection, Schmidt/Sherwood/Peclet numbers | `AdvectionDiffusionSolver` (needed) |
| 6 | Porous media transport, Darcy's law | `DarcyFlowSolver` (needed) |
| 7 | Exam 1 Review | - |
| 8-9 | Biofluids intro, Kinematics, Velocity/Acceleration | Velocity field utilities |
| 9-10 | Navier-Poisson, Newtonian/Non-Newtonian fluids | Constitutive model framework |
| 10-11 | Mass balance, Linear momentum balance | Core physics modules |
| 11-12 | Navier-Stokes equations | `NavierStokesSolver` (needed) |
| 12-13 | Euler/Bernoulli equations, NASA Bioreactor | Inviscid flow solvers |
| 13-14 | Poiseuille vs. Couette flow, Cylindrical flow | Pipe flow solvers |
| 14 | Viscoelasticity | Viscoelastic material models |

---

## 2. Part I: Mass Transport Fundamentals

### 2.1 Mathematical Preliminaries

#### 2.1.1 Scalar, Vector, and Tensor Quantities

**Definitions from HW1:**

| Quantity | Type | Example |
|----------|------|---------|
| Pressure | Scalar | p = -(σxx + σyy + σzz)/3 |
| Velocity | Vector | **v** = vx**î** + vy**ĵ** + vz**k̂** |
| Temperature | Scalar | T(x,y,z,t) |
| Distance | Scalar | d |
| Stress | Tensor | σij (9 components in 3D) |
| Acceleration | Vector | **a** = d**v**/dt |

**Implementation Note**: The library should provide clear type definitions distinguishing:
- Scalar fields (temperature, pressure, concentration)
- Vector fields (velocity, flux)
- Tensor fields (stress, deformation rate)

#### 2.1.2 Linear Algebra Operations

**Vector Operations (HW1 Problems 2a-2f):**

1. **Dot Product** (Orthogonality test):
   ```
   a · b = Σ(ai × bi) = |a||b|cos(θ)
   ```
   If a · b = 0, vectors are orthogonal.

2. **Cross Product** (Normal vector):
   ```
   a × b = |î  ĵ  k̂ |
           |a1 a2 a3|
           |b1 b2 b3|
   ```

3. **Tensor (Outer) Product**:
   ```
   a ⊗ b = [ai × bj] (3×3 matrix)
   ```

4. **Matrix Multiplication**: AB ≠ BA in general

5. **Determinant and Trace**:
   - det(A) = eigenvalue product
   - tr(A) = eigenvalue sum = Σ(Aii)

**Library Enhancement Needed**: Add utility functions for:
```cpp
namespace biotransport::linalg {
    double dot(const Vector3d& a, const Vector3d& b);
    Vector3d cross(const Vector3d& a, const Vector3d& b);
    Matrix3d outer(const Vector3d& a, const Vector3d& b);
    double determinant(const Matrix3d& A);
    double trace(const Matrix3d& A);
    std::vector<double> eigenvalues(const Matrix3d& A);
}
```

#### 2.1.3 Del Operator (∇)

The del operator is fundamental to all transport equations:

**Cartesian Coordinates:**
```
∇ = î(∂/∂x) + ĵ(∂/∂y) + k̂(∂/∂z)
```

**Key Operations:**
- **Gradient** (∇φ): scalar → vector
- **Divergence** (∇·**v**): vector → scalar  
- **Curl** (∇×**v**): vector → vector
- **Laplacian** (∇²φ = ∇·∇φ): scalar → scalar

**HW1 Problem 3 - Curl Example:**
```
v = (2z³ - y)î + (-4x + 7z)ĵ + (-3x)k̂

∇ × v = |î      ĵ      k̂    |
        |∂/∂x  ∂/∂y   ∂/∂z  |
        |2z³-y  -4x+7z  -3x  |

∇ × v = (0)î + (2)ĵ + (-4)k̂
```

**Implementation Status**: Current mesh classes support structured grids. Need:
- Gradient computation utilities
- Divergence computation utilities
- Curl computation for vorticity analysis

### 2.2 Diffusion Transport

#### 2.2.1 Fick's First Law

**Fundamental Form:**
```
ji = -Dim∇ρi
```

Where:
- **ji** = mass flux of species i [kg/(m²·s)]
- **Dim** = diffusion coefficient [m²/s]
- **ρi** = mass concentration [kg/m³]

**1D Steady-State Form (HW1 Problem 6):**
```
jd = -Dim × (ρ2 - ρ1)/(x2 - x1) = -Dim × Δρi/Δx
```

**Blood-Brain Barrier Example (HW1):**
- Drug concentration: 12 mg/L (blood) → 0.05 mg/L (brain)
- BBB thickness: 0.08 mm
- Diffusion coefficient: 1×10⁻⁹ m²/s

```
jd = -Dim × (ρbrain - ρblood)/L
jd = -(1×10⁻⁹) × (0.05 - 12)/0.00008
jd ≈ 0.1493 mg/(m²·s)
```

**Total Drug Transfer:**
```
m = jd × A × t
m = 0.1493 × (0.025×10⁻⁶) × (7×60)
m ≈ 1.57×10⁻⁶ mg
```

**Current Library Status**: `DiffusionSolver` implements this. ✓

#### 2.2.2 Mass and Molar Basis Conversions

**Key Relationships (HW1 Problem 5):**
```
ρi = Ci × Mi        (mass conc. = molar conc. × molecular weight)
ωi = ρi/ρ           (mass fraction)
χi = Ci/C           (mole fraction)
```

**Dilute Solution Limit**: ωi << 1 (typically < 0.01)

**Implementation Need**: Add conversion utilities:
```cpp
namespace biotransport::chemistry {
    double molarToMass(double C_molar, double molecular_weight);
    double massToMolar(double rho_mass, double molecular_weight);
    double massFraction(double rho_i, double rho_total);
    bool isDilute(double mass_fraction, double threshold = 0.01);
}
```

#### 2.2.3 Continuity Equation (Balance of Mass)

**General Form (HW1 Problem 7):**
```
∂ρ/∂t + ∇·(ρv) = 0
```

Or equivalently:
```
dρ/dt + ρ(∇·v) = 0
```

**Incompressible Flow (constant ρ):**
```
∇·v = 0
```

This is the **isochoric** (volume-preserving) condition for incompressible materials.

**Current Status**: Implicitly assumed in diffusion solvers. Need explicit incompressibility enforcement for Navier-Stokes.

### 2.3 Convection-Diffusion Transport

#### 2.3.1 Full Convection-Diffusion Equation

**General Form (HW2 Problem 3):**
```
∂ρi/∂t + v·∇ρi = Dim∇²ρi + ri
```

Where:
- Left side: local + convective change
- Right side: diffusion + reaction source/sink

**Simplifications:**

| Condition | Mathematical Form | Simplified Equation |
|-----------|-------------------|---------------------|
| No bulk motion | v·∇ρi = 0 | ∂ρi/∂t = Dim∇²ρi |
| Steady-state | ∂ρi/∂t = 0 | v·∇ρi = Dim∇²ρi + ri |
| SS + No motion | Both | Dim∇²ρi + ri = 0 |
| 1D SS Diffusion | Above + ∂/∂y = ∂/∂z = 0 | Dim(d²ρi/dx²) + ri = 0 |

**Current Library Status**: 
- Pure diffusion: ✓ `DiffusionSolver`
- Reaction-diffusion: ✓ `ReactionDiffusionSolver` variants
- Convection-diffusion: ✗ **NEEDED** - `AdvectionDiffusionSolver`

#### 2.3.2 Membrane Diffusion with Partition Coefficient

**Effective Flux (HW2 Problem 2):**
```
ji(x) = Dim × Φ × (ρA - ρB) / L
```

Where Φ is the partition coefficient describing solubility differences between phases.

**Hindered Diffusion (HW2 Problem 2):**
```
Dhindered = Dim × K₁(λ) × K₂(λ)
```

Where:
- λ = solute diameter / pore diameter
- K₁ = (1-λ)² (steric partition)
- K₂ = 1 - 2.014λ + 2.09λ³ - 0.95λ⁵ (hydrodynamic drag)

Valid for 0 ≤ λ ≤ 0.6.

**Hydrophobic Solute Behavior (HW2 Problem 5):**
For hydrophobic solutes in hydrophobic membranes:
- Concentration jumps UP at membrane interfaces
- Linear profile through membrane
- Results in enhanced transport

**Implementation Need**: 
```cpp
class MembraneDiffusionSolver {
public:
    void setPartitionCoefficient(double phi);
    void setHinderedDiffusion(double solute_diameter, double pore_diameter);
    // Handles concentration discontinuities at interfaces
};
```

### 2.4 Unsteady Diffusion

#### 2.4.1 Dimensionless Numbers

**Biot Modulus (HW2 Problem 6):**
```
Bi = hm × (V/A) / Di = hm × L_char / Di
```

Physical meaning: External mass transfer / Internal diffusion
- Bi << 1: Internal diffusion dominates → uniform concentration (lumped parameter OK)
- Bi >> 1: External resistance dominates

**Fourier Modulus:**
```
Fo = Di × t / L_char²
```

Physical meaning: Dimensionless time (diffusion penetration depth)

**Lumped Parameter Analysis (Bi << 1):**
```
(ρ - ρ∞)/(ρ₀ - ρ∞) = exp(-Bi × Fo)
```

**Microsphere Drug Release Example (HW2):**
- Diameter: 5 μm → L_char = r/3 = 0.833 μm
- hm = 2.0×10⁻⁵ m/s
- Di = 3.0×10⁻⁵ m²/s

```
Bi = (2.0×10⁻⁵)(2.5×10⁻⁶) / (3)(3.0×10⁻⁵) ≈ 5.56×10⁻⁴ << 1
```

Lumped OK! After 5 minutes:
```
Fo = (3.0×10⁻⁵)(300) / (8.33×10⁻⁷)² ≈ 1.30×10⁴

θ = exp(-Bi × Fo) ≈ exp(-7.22) ≈ 0
```

Drug completely released.

**Implementation Need**: Add dimensionless number calculators:
```cpp
namespace biotransport::dimensionless {
    double biot(double h_m, double L_char, double D);
    double fourier(double D, double t, double L_char);
    double peclet(double v, double L, double D);
    double reynolds(double rho, double v, double L, double mu);
    double schmidt(double nu, double D);  // nu = mu/rho
    double sherwood(double h_m, double L, double D);
}
```

### 2.5 Convection Mass Transfer

#### 2.5.1 Dimensionless Numbers for Convection

**From HW3:**

| Number | Formula | Physical Meaning |
|--------|---------|------------------|
| Reynolds (Re) | ρvL/μ | Inertia / Viscous forces |
| Schmidt (Sc) | μ/(ρDim) = ν/Dim | Momentum / Mass diffusivity |
| Peclet (Pe) | Re × Sc = vL/Dim | Convection / Diffusion rate |
| Sherwood (Sh) | hmL/Dim | Convective / Diffusive transport |

**Example (HW3 Problem 1) - Lymphatic Drug Transport:**
- Lymph density: 1005 kg/m³
- Vessel diameter: 50 μm
- Lymph viscosity: 1.23×10⁻³ Pa·s
- Lymph velocity: 1.5 mm/s
- Drug diffusion: 3.3×10⁻⁹ m²/s

```
Re = ρvD/μ = (1005)(0.0015)(50×10⁻⁶)/(1.23×10⁻³) ≈ 0.0613
Sc = μ/(ρD) = (1.23×10⁻³)/[(1005)(3.3×10⁻⁹)] ≈ 370
Pe = Re × Sc ≈ 22.7
```

Pe > 1 indicates convection dominates over diffusion.

#### 2.5.2 Concentration Boundary Layer

**Key Concept**: Near solid surfaces, concentration varies from surface value to bulk value over a thin "concentration boundary layer" δc(x).

**Implementation Need**: Boundary layer resolving mesh refinement utilities.

### 2.6 Buckingham Π-Theorem and Similitude

#### 2.6.1 Dimensional Analysis Procedure

**From HW3 Problem 3 - Pulse Wave Speed:**

Variables: c (wave speed), ρ (density), β (distensibility)

Step 1: Dimension matrix:
```
        c    ρ    β
   M [  0    1   -1  ]
   L [  1   -3    1  ]
   T [ -1    0    2  ]
```

Step 2: n = 3 variables, r = 3 independent dimensions → n - r = 0 independent Π groups

But we can form: Π₁ = c²ρβ

Result: c ∝ 1/√(ρβ) (Moens-Korteweg wave speed relation)

#### 2.6.2 Dynamic Similitude

**From HW3 Problem 4 - Scaled Vascular Model:**

For geometric scaling by factor S with fluid substitution:
```
Re_model = Re_actual

(ρv D/μ)_model = (ρv D/μ)_actual

v_model = v_actual × (ρ_a/ρ_m) × (D_a/D_m) × (μ_m/μ_a)
```

**Example**: 8× scaled Fontan graft model:
- Actual: D = 2 cm, v = 0.2 m/s, blood (ρ=1060, μ=0.0035)
- Model: D = 16 cm, water (ρ=1000, μ=0.001)

```
v_model = 0.2 × (1060/1000) × (2/16) × (0.001/0.0035)
v_model ≈ 0.0076 m/s
```

### 2.7 Transport in Porous Media

#### 2.7.1 Darcy's Law

**From HW3 Problem 5:**
```
q = -(κ/μ) ∇p
```

Where:
- q = volume flux (Darcy velocity) [m/s]
- κ = permeability [m²]
- μ = dynamic viscosity [Pa·s]
- ∇p = pressure gradient [Pa/m]

**Implications:**
- If κ decreases → flow rate decreases
- If |∇p| increases → flow rate increases

**Current Status**: Not implemented
**Need**: `DarcyFlowSolver` for tissue perfusion, tumor drug delivery enhancement

---

## 3. Part II: Fluid Mechanics and Biofluids

### 3.1 Kinematics

#### 3.1.1 Eulerian vs. Lagrangian Perspectives

**Lagrangian**: Follow individual fluid particles
- Position: **x**(t) of specific particle
- Velocity: **v** = d**x**/dt

**Eulerian**: Fixed observation points in space
- Velocity field: **v**(**x**, t)
- At any point, measure velocity of whatever particle is there

**Current Library Status**: Uses Eulerian (field) approach ✓

#### 3.1.2 Material Derivative (Eulerian Acceleration)

**From HW4 Problem 1d and HW4 Problem 4:**
```
Dv/Dt = ∂v/∂t + (v·∇)v
        ↑          ↑
      local    convective
   acceleration  acceleration
```

**Component Form:**
```
ai = ∂vi/∂t + vx(∂vi/∂x) + vy(∂vi/∂y) + vz(∂vi/∂z)
```

**Example (HW4 Problem 4):**
For v = (xt + 2y)î + (zyt² - yt)ĵ:

1. **Steady?** ∂v/∂t = xî + (2zyt - y)ĵ ≠ 0 → Not steady

2. **Incompressible?** ∇·v = t + (zt² - t) = zt² ≠ 0 → Compressible

3. **Acceleration:**
```
ax = x + (xt + 2y)t + 2(zyt² - yt)
ay = 2zyt - y + (zyt² - yt)(zt² - t)
az = 0
```

4. **Irrotational?** ∇×v = -yt²î - 2k̂ ≠ 0 → Rotational flow

**Implementation Need**: Material derivative calculator for velocity fields.

### 3.2 Cauchy Stress Tensor

#### 3.2.1 Stress Tensor Components

**From HW4 Problem 1b:**

The Cauchy stress tensor σ:
```
σ = | σxx  σxy  σxz |
    | σyx  σyy  σyz |
    | σzx  σzy  σzz |
```

**Physical Interpretation of σij:**
- i = face normal direction
- j = force direction
- Diagonal (σii): Normal stresses
- Off-diagonal (σij, i≠j): Shear stresses

**Pressure (Negative Mean Normal Stress):**
```
p = -(σxx + σyy + σzz)/3
```

#### 3.2.2 Eigenvalues and Principal Stresses

**From HW4 Problem 2a:**

For σ:
```
σ = | 2  2  0 |
    | 2  4  0 |
    | 0  0  1 |
```

Characteristic equation: det(σ - λI) = 0
```
(1-λ)(λ² - 6λ + 4) = 0
```

Eigenvalues (principal stresses): λ₁ = 1, λ₂ = 3-√5, λ₃ = 3+√5

Pressure: p = -(2 + 4 + 1)/3 = -7/3 Pa

**Key Property**: Pressure is rotation invariant (proven in HW4 Problem 2c).

### 3.3 Deformation Rate Tensor

#### 3.3.1 Definition and Components

**From HW5 Problem 2:**

The deformation rate tensor D:
```
Dij = (1/2)(∂vj/∂i + ∂vi/∂j)
```

**Diagonal Components** (extension rates):
```
Dxx = ∂vx/∂x,  Dyy = ∂vy/∂y,  Dzz = ∂vz/∂z
```

**Off-diagonal Components** (shear rates):
```
Dxy = Dyx = (1/2)(∂vy/∂x + ∂vx/∂y)
Dyz = Dzy = (1/2)(∂vz/∂y + ∂vy/∂z)
Dxz = Dzx = (1/2)(∂vz/∂x + ∂vx/∂z)
```

**Cylindrical Coordinates (HW5 Problem 2c):**

For axial flow v = vz(r)êz:
```
Drr = ∂vr/∂r = 0
Drz = (1/2)(∂vr/∂z + ∂vz/∂r) = (1/2)(∂vz/∂r)
```

### 3.4 Constitutive Equations

#### 3.4.1 Newtonian Fluids

**Definition (HW5 Problem 1b):** Isotropic fluid with linear relationship between shear stress and deformation rate.

**Navier-Poisson Equations (HW4 Problem 5):**
```
σii = -p + λ(∇·v) + 2μDii        (normal stresses)
σij = 2μDij, i ≠ j               (shear stresses)
```

Where:
- μ = dynamic viscosity [Pa·s]
- λ = second viscosity coefficient

**For Incompressible Flow (∇·v = 0):**
```
σii = -p + 2μDii
σij = 2μDij, i ≠ j
```

#### 3.4.2 Non-Newtonian Fluids (Power Law)

**From Exam Equation Sheet:**
```
σij = k|∂vj/∂i|^(n-1) × (∂vj/∂i)
```

Where:
- k = consistency index
- n = power law index
  - n < 1: Shear thinning (blood)
  - n = 1: Newtonian
  - n > 1: Shear thickening

**Implementation Need**: 
```cpp
class ConstitutiveModel {
public:
    virtual Tensor3d stress(const Tensor3d& D, double p) = 0;
};

class NewtonianFluid : public ConstitutiveModel { /* ... */ };
class PowerLawFluid : public ConstitutiveModel { /* ... */ };
class CarreauFluid : public ConstitutiveModel { /* ... */ };  // For blood
```

### 3.5 Fundamental Balance Relations

#### 3.5.1 Mass Balance (Continuity)

**Differential Form (HW5 Problem 3a):**
```
∂ρ/∂t + ∇·(ρv) = 0
```

Equivalent form using material derivative:
```
dρ/dt + ρ(∇·v) = 0
```

**Proof** (from HW5 Problem 3a):

Starting with dρ/dt + ρ(∇·v) = 0 and expanding:
```
∂ρ/∂t + vx(∂ρ/∂x) + vy(∂ρ/∂y) + vz(∂ρ/∂z) + ρ(∂vx/∂x + ∂vy/∂y + ∂vz/∂z) = 0

∂ρ/∂t + ∂(ρvx)/∂x + ∂(ρvy)/∂y + ∂(ρvz)/∂z = 0

∂ρ/∂t + ∇·(ρv) = 0  ✓
```

#### 3.5.2 Linear Momentum Balance

**General Form:**
```
∂σxi/∂x + ∂σyi/∂y + ∂σzi/∂z + ρgi = ρai
```

**Component Equations:**
```
x: ∂σxx/∂x + ∂σyx/∂y + ∂σzx/∂z + ρgx = ρax
y: ∂σxy/∂x + ∂σyy/∂y + ∂σzy/∂z + ρgy = ρay  
z: ∂σxz/∂x + ∂σyz/∂y + ∂σzz/∂z + ρgz = ρaz
```

**Derivation Approach** (HW5 Problem 3b): Force balance on infinitesimal fluid element, summing stress contributions on all faces plus body forces.

### 3.6 Navier-Stokes Equations

#### 3.6.1 Incompressible Newtonian Form

**Vector Form (HW5 Problem 5a):**
```
-∇p + μ∇²v + ρg = ρ(∂v/∂t + (v·∇)v)
```

**Component Form (z-direction from HW5 Problem 5a):**
```
-∂p/∂z + μ∇²vz + ρgz = ρ(∂vz/∂t + (v·∇)vz)
```

Where:
- -∇p: Pressure gradient (driving force)
- μ∇²v: Viscous diffusion
- ρg: Body force (gravity)
- ρ∂v/∂t: Local acceleration
- ρ(v·∇)v: Convective acceleration

#### 3.6.2 Fluid Statics

**From HW5 Problem 5b and HW6 Problem 2b:**

For static flow (v = 0, a = 0):
```
-∇p + ρg = 0
∇p = ρg
```

In a beaker with z pointing up (g = -gz k̂):
```
∂p/∂x = 0
∂p/∂y = 0  
∂p/∂z = -ρgz
```

Integrating: **p(z) = p₀ - ρgzz**

At depth D below surface: **p = patm + ρgzD**

#### 3.6.3 Cylindrical Navier-Stokes

**From Navier-Stokes Worksheets:**

**Mass Balance:**
```
(1/r)∂(rvr)/∂r + (1/r)∂vθ/∂θ + ∂vz/∂z = 0
```

**r-Momentum:**
```
-∂p/∂r + μ[∂/∂r((1/r)∂(rvr)/∂r) + (1/r²)∂²vr/∂θ² - (2/r²)∂vθ/∂θ + ∂²vr/∂z²] + ρgr
= ρ(∂vr/∂t + vr∂vr/∂r + (vθ/r)∂vr/∂θ - vθ²/r + vz∂vr/∂z)
```

**θ-Momentum:**
```
-(1/r)∂p/∂θ + μ[∂/∂r((1/r)∂(rvθ)/∂r) + (1/r²)∂²vθ/∂θ² + (2/r²)∂vr/∂θ + ∂²vθ/∂z²] + ρgθ
= ρ(∂vθ/∂t + vr∂vθ/∂r + (vθ/r)∂vθ/∂θ + vrvθ/r + vz∂vθ/∂z)
```

**z-Momentum:**
```
-∂p/∂z + μ[(1/r)∂/∂r(r∂vz/∂r) + (1/r²)∂²vz/∂θ² + ∂²vz/∂z²] + ρgz
= ρ(∂vz/∂t + vr∂vz/∂r + (vθ/r)∂vz/∂θ + vz∂vz/∂z)
```

**Implementation Need**: Full cylindrical coordinate support for pipe flow problems.

### 3.7 Special Flow Solutions

#### 3.7.1 Couette Flow (Parallel Plates)

**From HW6 Problem 2c - Pressure-Driven Flow:**

**Assumptions:**
1. Newtonian (μ constant)
2. Incompressible (∇·v = 0)
3. Steady (∂v/∂t = 0)
4. Unidirectional (vy = vz = 0)
5. Negligible body forces (g = 0)
6. Fully developed (∂v/∂x = 0)
7. 1-D flow (∂vx/∂x = ∂vx/∂z = 0)

**Reduced Equations:**
```
Mass: 0 = 0 (satisfied)
x-Mom: μ(∂²vx/∂y²) = ∂p/∂x
y-Mom: 0 = ∂p/∂y
z-Mom: 0 = ∂p/∂z
```

**Solution (origin at centerline, y ∈ [-h/2, h/2]):**
```
vx(y) = (1/2μ)(dp/dx)(h²/4 - y²)
```

**Maximum Velocity (at centerline):**
```
vmax = (h²/8μ)(-dp/dx)
```

**Flow Rate:**
```
Q = (h³w/12μ)(-dp/dx)
```

**Wall Shear Stress:**
```
τw = |μ(∂vx/∂y)|_{y=±h/2} = (h/2)|dp/dx|
```

#### 3.7.2 Poiseuille Flow (Cylindrical Pipe)

**From Poiseuille Practice Problem:**

**Assumptions:**
1. Newtonian
2. Incompressible
3. Steady
4. Axial flow only (vr = vθ = 0)
5. Fully developed (∂v/∂z = 0)
6. Axisymmetric (∂v/∂θ = 0)
7. Negligible body forces

**Reduced Equations:**
```
Mass: 0 = 0 (satisfied)
r-Mom: -∂p/∂r = 0
θ-Mom: -∂p/∂θ = 0
z-Mom: -∂p/∂z + μ[(1/r)∂/∂r(r∂vz/∂r)] = 0
```

**ODE to Solve:**
```
(1/μ)(dp/dz) = (1/r)d/dr(r·dvz/dr) = d²vz/dr² + (1/r)dvz/dr
```

**Solution Approach:**

Assume vz = kr^n for homogeneous part:
```
n²r^(n-2) = (n/r²)r^n → vh = k (constant)
```

Particular solution vp = Ar + Br²:
```
0 + 4B = (1/μ)(dp/dz) → B = (1/4μ)(dp/dz), A = 0
```

**General Solution:**
```
vz(r) = (1/4μ)(dp/dz)r² + C₁ln(r) + C₂
```

**Boundary Conditions:**
- Finite at r = 0 → C₁ = 0
- No-slip at r = a → vz(a) = 0

**Final Solution:**
```
vz(r) = (a²/4μ)(-dp/dz)(1 - r²/a²)
```

**Hagen-Poiseuille Equation (Flow Rate):**
```
Q = πa⁴/(8μ) × (-dp/dz)
```

#### 3.7.3 NASA Bioreactor (Rotating Cylinders)

**From NASA Bioreactor Practice Problem:**

**Setup:** Concentric cylinders at r = a (inner, ωa) and r = b (outer, ωb)

**Assumptions:** Same as Poiseuille plus axisymmetric, negligible body forces

**Velocity Field:** v = vr(r)êr + vθ(r)êθ

**From Mass Balance:**
```
(1/r)∂(rvr)/∂r = 0 → rvr = C₁ → vr = C₁/r
```

With no-penetration at walls: vr(a) = vr(b) = 0 → **vr = 0**

**Reduced Momentum:**
```
r: -∂p/∂r = -ρvθ²/r
θ: μ∂/∂r((1/r)∂(rvθ)/∂r) = 0
```

**Solving θ-equation:**
```
(1/r)d(rvθ)/dr = C₂
rvθ = C₂r²/2 + C₃
vθ = (C₂/2)r + C₃/r
```

**Boundary Conditions:**
- vθ(a) = aωa
- vθ(b) = bωb

**Solution:**
```
vθ(r) = ((b²ωb - a²ωa)/(b² - a²))r + (a²b²(ωa - ωb)/(b² - a²))(1/r)
```

**Implementation Need**: 
```cpp
class CouetteFlowSolver {
    // Planar: parallel plates
    // Cylindrical: concentric rotating cylinders
};

class PoiseuilleFlowSolver {
    // Pressure-driven pipe flow
};
```

### 3.8 Euler and Bernoulli Equations

#### 3.8.1 Euler Equation (Inviscid Flow)

**From μ = 0 in Navier-Stokes:**
```
-∇p + ρg = ρ(∂v/∂t + (v·∇)v)
```

#### 3.8.2 Bernoulli Equation

**From HW6 Problem 4:**

For steady, incompressible, inviscid, irrotational flow along a streamline:
```
p + ρgz + (1/2)ρv² = constant
```

Or between two points:
```
p₁ + ρgz₁ + (1/2)ρv₁² = p₂ + ρgz₂ + (1/2)ρv₂²
```

**Applicability Check (HW6 Problem 4b):**

For v = (x+y)î + (x-y)ĵ:
1. Steady? ∂v/∂t = 0 ✓
2. Incompressible? ∇·v = 1 + (-1) = 0 ✓
3. Irrotational? ∇×v = (1-1)k̂ = 0 ✓

→ Bernoulli can be used!

**Flow Device Example (HW6 Problem 4a):**

Given: Same pressure gauges, D₁ = 2cm, v₁ = 10 cm/s, h = 10cm

With p₁ = p₂:
```
ρgz₁ + (1/2)ρv₁² = ρgz₂ + (1/2)ρv₂²
v₂ = √(2g(z₁-z₂) + v₁²) = √(2(9.81)(0.1) + 0.1²) ≈ 1.4 m/s
```

**Venturi Meter (HW6 Problem 4c):**

Combined with mass balance A₁v₁ = A₂v₂:
```
v₁ = √(2(ρHg - ρf)gΔh / (ρf[(A₁/A₂)² - 1]))
```

### 3.9 Viscoelasticity

#### 3.9.1 Basic Models

**From HW6 Problem 5:**

**Spring (Elastic):** σ = kε

**Dashpot (Viscous):** σ = μ(dε/dt)

**Maxwell Model** (spring + dashpot in series):
- Stress relaxation
- Flow under constant stress

**Kelvin-Voigt Model** (spring + dashpot in parallel):
- Creep under constant stress
- No stress relaxation

#### 3.9.2 Standard Linear Solid

**Governing Equation (HW6 Problem 5a):**
```
dσ/dt + (E₂/μ)σ = (E₂E₁/μ)ε + (E₂+E₁)(dε/dt)
```

**Stress Relaxation Response:**

For ε(t) = ε₀ (constant strain):
```
dσ/dt + (E₂/μ)σ = (E₂E₁/μ)ε₀
```

Solution:
```
σ(t) = ε₀(E₁ + E₂·exp(-E₂t/μ))
```

At t = 0: σ(0) = ε₀(E₁ + E₂) (both springs contribute)
At t → ∞: σ(∞) = ε₀E₁ (dashpot fully relaxed)

#### 3.9.3 Burgers Model (4-Parameter)

**Creep Function (HW6 Problem 5b):**
```
J(t) = ε(t)/σ₀ = 1/E₁ + t/μ₁ + (1/E₂)(1 - exp(-E₂t/μ₂))
```

Components:
- 1/E₁: Instantaneous elastic strain
- t/μ₁: Viscous flow (permanent)
- (1/E₂)(1 - exp(-E₂t/μ₂)): Delayed elastic recovery

#### 3.9.4 Complex Modulus (Dynamic Testing)

**For Sinusoidal Strain (HW6 Problem 5c):**
```
ε(t) = εA·sin(ωt)
σ(t) = σA·sin(ωt + φ)
```

Complex modulus:
```
σ/ε = (σA/εA)·exp(iφ) = G₁ + iG₂
```

Where:
- G₁ = (σA/εA)cos(φ) = Storage modulus (elastic part)
- G₂ = (σA/εA)sin(φ) = Loss modulus (viscous part)
- tan(φ) = G₂/G₁ = Loss tangent

**Implementation Need**:
```cpp
class ViscoelasticModel {
public:
    virtual double stress(double strain, double strain_rate, double t) = 0;
    virtual double stressRelaxation(double epsilon_0, double t) = 0;
    virtual double creep(double sigma_0, double t) = 0;
};

class MaxwellModel : public ViscoelasticModel { /* ... */ };
class KelvinVoigtModel : public ViscoelasticModel { /* ... */ };
class StandardLinearSolid : public ViscoelasticModel { /* ... */ };
class BurgersModel : public ViscoelasticModel { /* ... */ };
```

---

## 4. Homework Problem Analysis

### 4.1 HW1: Foundations

| Problem | Topic | Key Concepts | Library Relevance |
|---------|-------|--------------|-------------------|
| 1 | Scalars/Vectors/Tensors | Type classification | Type system design |
| 2a-f | Linear Algebra | Dot, cross, outer products, eigenvalues | Math utilities |
| 3 | Del Operator | Curl calculation | Differential operators |
| 4 | Mass Transport Concepts | Diffusion vs. convection | Solver selection |
| 5 | Concentration Units | Mass/molar conversion | Unit handling |
| 6 | Fick's Law | 1D diffusion flux | `DiffusionSolver` |
| 7 | Continuity Equation | Incompressibility | Mass conservation |

### 4.2 HW2: Diffusion Details

| Problem | Topic | Key Concepts | Library Relevance |
|---------|-------|--------------|-------------------|
| 1 | Membrane Diffusion | Fick's law, flux, rate | Basic diffusion |
| 2 | Hindered Diffusion | Partition coefficient | `MembraneDiffusionSolver` |
| 3-4 | Convection-Diffusion | Equation simplification | Equation framework |
| 5 | Partition Effects | Interface concentration jumps | Interface handling |
| 6 | Unsteady Diffusion | Bi, Fo, lumped analysis | Time-stepping |

### 4.3 HW3: Dimensionless Analysis

| Problem | Topic | Key Concepts | Library Relevance |
|---------|-------|--------------|-------------------|
| 1 | Peclet Number | Re, Sc, Pe calculation | Dimensionless utilities |
| 2 | Boundary Layer | Concentration BL concept | Mesh refinement |
| 3 | Π-Theorem | Dimensional analysis | Problem scaling |
| 4 | Similitude | Model scaling | Validation tools |
| 5 | Darcy's Law | Porous media flow | `DarcyFlowSolver` |

### 4.4 HW4: Kinematics & Stress

| Problem | Topic | Key Concepts | Library Relevance |
|---------|-------|--------------|-------------------|
| 1 | Definitions | Fluid, pressure, stress | Glossary/types |
| 2 | Cauchy Stress | Eigenvalues, pressure | Stress analysis |
| 3 | Cylindrical ∇ | Divergence derivation | Coordinate support |
| 4 | Velocity Analysis | Steady, incomp., irrotational | Flow classification |
| 5 | Navier-Poisson | Constitutive equations | Material models |

### 4.5 HW5: Balance Laws

| Problem | Topic | Key Concepts | Library Relevance |
|---------|-------|--------------|-------------------|
| 1 | Definitions | Viscosity, Newtonian, no-slip | Boundary conditions |
| 2 | Deformation Rate | D tensor components | Strain rate utilities |
| 3 | Balance Laws | Mass & momentum derivation | Core physics |
| 4 | Bioreactor Stress | Cauchy stress application | Rotating flow |
| 5 | Navier-Stokes | N-S derivation, statics | Core solver |

### 4.6 HW6: Flow Solutions

| Problem | Topic | Key Concepts | Library Relevance |
|---------|-------|--------------|-------------------|
| 1 | Definitions | Steady, axisymmetric, etc. | Problem classification |
| 2 | N-S Practice | Derivation, static pressure | Solver framework |
| 3 | N-S Cylindrical | Pipe flow reduction | Cylindrical coords |
| 4 | Bernoulli | Inviscid analysis | Simple flow tools |
| 5 | Viscoelasticity | SLS, Burgers, complex mod. | Material models |

---

## 5. Practice Problem Deep Dives

### 5.1 NASA Bioreactor Problem

**Physical Setup:**
- Two concentric cylinders
- Inner (r = a): angular velocity ωa
- Outer (r = b): angular velocity ωb
- Newtonian fluid between cylinders

**Solution Procedure:**

**Step 1: State Assumptions**
1. Newtonian (μ constant)
2. Incompressible (∇·v = 0)
3. Steady (∂v/∂t = 0)
4. No axial flow (vz = 0)
5. Fully developed (∂v/∂z = 0)
6. Axisymmetric (∂v/∂θ = 0)
7. Negligible body forces (g = 0)

**Step 2: Apply Mass Balance**
```
(1/r)∂(rvr)/∂r + (1/r)∂vθ/∂θ + ∂vz/∂z = 0
      ↓              ↓           ↓
   keep            =0(6)       =0(4)

(1/r)d(rvr)/dr = 0 → rvr = C → vr = C/r
```

**Step 3: Apply Boundary Conditions (No Penetration)**
```
vr(a) = 0, vr(b) = 0 → C = 0 → vr = 0
```

**Step 4: Reduce Navier-Stokes**
```
r-momentum: -∂p/∂r = -ρvθ²/r    (pressure balances centrifugal)
θ-momentum: μ∂/∂r[(1/r)∂(rvθ)/∂r] = 0
z-momentum: 0 = 0
```

**Step 5: Solve θ-Equation**
```
∂/∂r[(1/r)∂(rvθ)/∂r] = 0 → (1/r)d(rvθ)/dr = C₂

Integrating: rvθ = C₂r²/2 + C₃

vθ(r) = (C₂/2)r + C₃/r
```

**Step 6: Apply No-Slip Conditions**
```
vθ(a) = aωa → (C₂/2)a + C₃/a = aωa
vθ(b) = bωb → (C₂/2)b + C₃/b = bωb
```

Solving system:
```
C₂ = 2(b²ωb - a²ωa)/(b² - a²)
C₃ = a²b²(ωa - ωb)/(b² - a²)
```

**Final Velocity Profile:**
```
vθ(r) = ((b²ωb - a²ωa)/(b² - a²))r + (a²b²(ωa - ωb)/(b² - a²))(1/r)
```

**Applications:**
1. **Viscometer**: Inner cylinder rotates, measure torque → calculate viscosity
2. **NASA Bioreactor**: Simulate microgravity by balancing centrifugal force

**Implementation Need:**
```cpp
class TaylorCouetteFlowSolver {
public:
    struct Config {
        double inner_radius;
        double outer_radius;
        double inner_omega;
        double outer_omega;
        double viscosity;
        double density;
    };
    
    std::vector<double> velocityProfile(int num_points) const;
    std::vector<double> pressureProfile(int num_points) const;
    double torquePerLength() const;
    double apparentViscosity() const;  // For viscometer application
};
```

### 5.2 Poiseuille Flow Problem

**Physical Setup:**
- Cylindrical pipe, radius a
- Pressure-driven flow
- No rotation

**Solution Summary:**

**Velocity:**
```
vz(r) = (a²/4μ)(-dp/dz)(1 - r²/a²) = vmax(1 - r²/a²)
```

Where vmax = a²(-dp/dz)/(4μ)

**Flow Rate:**
```
Q = ∫₀ᵃ vz(r) · 2πr dr = (πa⁴/8μ)(-dp/dz)
```

**Wall Shear Stress:**
```
τw = μ|dvz/dr|_{r=a} = (a/2)(-dp/dz)
```

**Mean Velocity:**
```
v̄ = Q/(πa²) = (a²/8μ)(-dp/dz) = vmax/2
```

**Pressure Drop:**
```
Δp = (8μLQ)/(πa⁴) = (128μLQ)/(πD⁴)
```

**Implementation Need:**
```cpp
class HagenPoiseuilleFlowSolver {
public:
    struct Config {
        double radius;
        double length;
        double pressure_drop;  // or flow_rate
        double viscosity;
    };
    
    double maxVelocity() const;
    double meanVelocity() const;
    double flowRate() const;
    double wallShearStress() const;
    double pressureDrop() const;
    std::vector<double> velocityProfile(int num_points) const;
};
```

---

## 6. Core Equations Reference

### 6.1 Exam 2 Equation Sheet Analysis

The following equations are provided on exams and represent core knowledge:

**Kinematics:**
```
u(t) = ux(t)î + uy(t)ĵ + uz(t)k̂
dv/dt = ∂v/∂t + (v·∇)v
```

**Eigenvalue Problem:**
```
Ax = λx ⟺ |A - λI| = 0
```

**Euler's Formula:**
```
e^(iθ) = cos(θ) + i·sin(θ)
```

**Differentiation:**
```
d/dx[b^(kx)] = kb^(kx)·ln(b)
```

**Navier-Poisson Equations:**
```
σii = -p + λ(∇·v) + 2μDii
σij = 2μDij, i ≠ j
```

**Power Law Fluid:**
```
σij = k|∂vj/∂i|^(n-1) · (∂vj/∂i)
```

**Deformation Rate:**
```
Dij = (1/2)(∂vj/∂i + ∂vi/∂j)
```

**Vorticity:**
```
ξ = ∇ × v
```

**Pressure:**
```
p = -(1/3)(σxx + σyy + σzz)
```

**Continuity:**
```
dm = ρdV; dV = dx dy dz
dρ/dt + ρ(∇·v) = 0
```

**Momentum Balance:**
```
∂σxi/∂x + ∂σyi/∂y + ∂σzi/∂z + ρgi = ρai
```

**Navier-Stokes:**
```
-∇p + μ∇²v + ρg = ρa
```

**Bernoulli:**
```
p + ρgz + (1/2)ρv² = C
```

**Mass Balance:**
```
A₁v₁ = A₂v₂
Q = ∫_A v_normal dA
v̄ = Q/A
```

**Wall Shear:**
```
τw = |σij|_wall = 2μ|Dij|_wall
σrz = k|dvz/dr|^n
```

**Elasticity:**
```
σ = kε        (Hooke's law)
σ = μ(dε/dt)  (Viscous element)
```

---

## 7. Implementation Gap Analysis

### 7.1 Current Library Capabilities

Based on the BioTransport library documentation provided:

**Implemented (✓):**
- `StructuredMesh` (1D and 2D)
- `DiffusionSolver` (basic diffusion)
- `ReactionDiffusionSolver` (callback-based)
- `LinearReactionDiffusionSolver` (decay)
- `LogisticReactionDiffusionSolver` (growth with carrying capacity)
- `MichaelisMentenReactionDiffusionSolver` (enzyme kinetics)
- `ConstantSourceReactionDiffusionSolver` (source/sink)
- `VariableDiffusionSolver` (spatially varying D)
- `MaskedMichaelisMentenReactionDiffusionSolver` (regional reactions)
- `GrayScottSolver` (two-species pattern formation)
- `TumorDrugDeliverySolver` (multi-physics)
- `BioheatCryotherapySolver` (thermal + damage)
- `ExplicitFD` facade with stable dt selection
- Boundary conditions (Dirichlet, Neumann)

### 7.2 Gaps Identified from BMEN 341

**High Priority (Core Course Content):**

| Gap | Course Topics | Suggested Implementation |
|-----|---------------|-------------------------|
| **Convection-Diffusion** | Weeks 5, HW3 | `AdvectionDiffusionSolver` |
| **Darcy Flow** | Week 6, HW3.5 | `DarcyFlowSolver` |
| **Navier-Stokes** | Weeks 11-13 | `NavierStokesSolver` |
| **Cylindrical Coordinates** | HW3-6, Practice | Mesh and operator extensions |
| **Non-Newtonian Fluids** | Week 9-10 | `PowerLawFluidModel` |

**Medium Priority (Course Supporting Content):**

| Gap | Course Topics | Suggested Implementation |
|-----|---------------|-------------------------|
| **Viscoelasticity** | Week 14 | `ViscoelasticMaterialModel` |
| **Membrane Transport** | Weeks 2-3 | `MembraneDiffusionSolver` with partition |
| **Dimensionless Numbers** | HW3 | `DimensionlessCalculator` utilities |
| **Pipe Flow Solutions** | Week 13-14 | `PoiseuilleFlowSolver`, `CouetteFlowSolver` |
| **Bernoulli Solver** | Week 12 | `BernoulliEquationSolver` |

**Lower Priority (Educational/Verification):**

| Gap | Course Topics | Suggested Implementation |
|-----|---------------|-------------------------|
| **Analytical Solutions** | All HW | Verification test suite |
| **Dimensional Analysis** | HW3 | `PiTheoremHelper` |
| **Similitude Scaling** | HW3 | `SimilitudeScaler` |
| **Vector/Tensor Utilities** | HW1, HW4 | Extended `linalg` namespace |

### 7.3 Detailed Gap Specifications

#### 7.3.1 AdvectionDiffusionSolver

**Physics:**
```
∂c/∂t + v·∇c = D∇²c + r
```

**Requirements:**
- Velocity field input (could be from Darcy, Stokes, or N-S)
- Upwind schemes for convection stability
- Peclet-aware mesh refinement or stabilization
- Option for SUPG stabilization

**API Sketch:**
```cpp
class AdvectionDiffusionSolver {
public:
    AdvectionDiffusionSolver(const StructuredMesh& mesh, 
                              double diffusivity,
                              std::function<Vector3d(double x, double y)> velocity);
    
    void setReactionTerm(std::function<double(double c)> reaction);
    void solve(double dt, int num_steps);
    double pecletNumber() const;
    bool isConvectionDominated() const;
};
```

#### 7.3.2 DarcyFlowSolver

**Physics:**
```
v = -(κ/μ)∇p
∇·v = 0  (for incompressible)
```

**Requirements:**
- Permeability field (scalar or tensor)
- Pressure boundary conditions
- Solve Laplace/Poisson equation for pressure
- Compute velocity field from pressure gradient

**Integration with Current Library:**
- `TumorDrugDeliverySolver` already has `solvePressureSOR()` - extract and generalize

**API Sketch:**
```cpp
class DarcyFlowSolver {
public:
    DarcyFlowSolver(const StructuredMesh& mesh,
                    const std::vector<double>& permeability,
                    double viscosity);
    
    void setPressureBoundary(Boundary b, double p);
    void setFluxBoundary(Boundary b, double flux);
    void solve();
    
    const std::vector<double>& pressure() const;
    const std::vector<double>& velocityX() const;
    const std::vector<double>& velocityY() const;
};
```

#### 7.3.3 StokesSolver (Precursor to Navier-Stokes)

**Physics (Steady Stokes):**
```
-∇p + μ∇²v = 0
∇·v = 0
```

**Requirements:**
- Coupled velocity-pressure solve
- Saddle-point system handling
- Staggered or stabilized equal-order elements

**API Sketch:**
```cpp
class StokesSolver {
public:
    StokesSolver(const StructuredMesh& mesh, double viscosity);
    
    void setVelocityBoundary(Boundary b, Vector2d v);
    void setPressureBoundary(Boundary b, double p);
    void setBodyForce(std::function<Vector2d(double x, double y)> f);
    void solve();
    
    const std::vector<double>& velocityX() const;
    const std::vector<double>& velocityY() const;
    const std::vector<double>& pressure() const;
};
```

#### 7.3.4 NavierStokesSolver

**Physics (Incompressible):**
```
ρ(∂v/∂t + v·∇v) = -∇p + μ∇²v + ρg
∇·v = 0
```

**Complexity**: This is a major undertaking requiring:
- Projection methods or coupled solvers
- Nonlinear convective term handling
- Pressure-velocity coupling
- Time integration for transient

**Recommendation**: Start with:
1. Steady Stokes (linear, decoupled-ish)
2. Time-dependent Stokes  
3. Full Navier-Stokes

#### 7.3.5 CylindricalMesh

**Requirements:**
- Support for (r, θ, z) coordinates
- Proper handling of r = 0 singularity
- Operators (gradient, divergence, Laplacian) in cylindrical form

**API Sketch:**
```cpp
class CylindricalMesh {
public:
    CylindricalMesh(int nr, int ntheta, int nz,
                    double rmin, double rmax,
                    double thetamin, double thetamax,
                    double zmin, double zmax);
    
    double r(int i) const;
    double theta(int j) const;
    double z(int k) const;
    int index(int i, int j, int k) const;
    
    // Handle r=0 singularity
    bool hasAxisSingularity() const;
};
```

#### 7.3.6 AnalyticalSolutions Namespace

**Purpose**: Verification against known solutions from course

```cpp
namespace biotransport::analytical {

// HW2-style: Unsteady diffusion
double diffusion1D_semiInfinite(double x, double t, double D, 
                                 double C0, double Cinf);

// HW6-style: Poiseuille flow
double poiseuille_velocity(double r, double a, double dpdz, double mu);
double poiseuille_flowRate(double a, double dpdz, double mu);

// NASA Bioreactor: Taylor-Couette
double taylorCouette_velocity(double r, double a, double b,
                               double omega_a, double omega_b);

// Couette flow between parallel plates
double couette_velocity(double y, double h, double dpdx, double mu);

// Bernoulli calculations
double bernoulli_velocity(double p1, double p2, double z1, double z2,
                          double v1, double rho);

// Viscoelastic responses
double maxwellRelaxation(double E, double eta, double epsilon0, double t);
double kelvinVoigtCreep(double E, double eta, double sigma0, double t);
double SLSRelaxation(double E1, double E2, double eta, double epsilon0, double t);
double burgersCreep(double E1, double mu1, double E2, double mu2, 
                     double sigma0, double t);

}
```

---

## 8. Library Extensions (Status)

The “recommended extensions” in earlier iterations of this document have been implemented in the library.

Implemented items (course-aligned):
- Dimensionless number utilities + stability helpers (Pe/CFL/Fo; dt suggestion helpers)
- AdvectionDiffusionSolver (upwind + explicit stability considerations)
- Membrane transport support (partitioning/interface behavior)
- Analytical solution suite for canonical validation
- DarcyFlowSolver as a first-class flow building block

For the current API surface and include paths, see docs/notes/FOOTPRINT.md.

2. **StokesSolver**
   - Steady viscous flow
   - Foundation for Navier-Stokes

3. **Cylindrical Coordinate Support**
   - CylindricalMesh class
   - Cylindrical differential operators
   - Pipe flow examples

4. **Constitutive Model Framework**
   - Abstract base class
   - Newtonian implementation
   - Power-law implementation

### 8.3 Long-Term Goals

1. **Full Navier-Stokes Solver**
   - Projection methods
   - Transient capability
   - Non-Newtonian extensions

2. **Viscoelastic Material Library**
   - Maxwell, Kelvin-Voigt, SLS, Burgers models
   - Time-domain response functions
   - Frequency-domain (complex modulus)

3. **Coupled Multi-Physics**
   - Fluid-structure interaction basics
   - Heat + mass + momentum coupling

### 8.4 Documentation Improvements

Based on the course structure:

1. **Theory Guide**: Add sections mirroring course topics
   - Part I: Mass Transport
   - Part II: Fluid Mechanics
   
2. **Example Problems**: Implement homework-style examples
   - Drug diffusion across BBB
   - Microsphere release kinetics
   - Pipe flow analysis
   - Rotating cylinder viscometer

3. **Verification Tests**: Course problems as test cases
   - Analytical comparisons
   - Conservation checks
   - Convergence studies

---

## 9. Summary (Status)

BMEN 341 topic coverage is complete in the current library.

- Dimensionless groups and analytical references exist in C++ (and are exposed to Python where appropriate).
- Core PDE solvers for the course (diffusion, reaction-diffusion, advection–diffusion, Darcy flow, membrane diffusion) are implemented.
- Time-step stability helpers are available (see the stability utilities documented in `docs/notes/FOOTPRINT.md`).

No open action items remain from this analysis. For any future/optional extensions, use `docs/notes/ROADMAP.md` as the living planning document.

---

## Appendix A: Course Problem Index

| HW | Problem | Topic | Key Equation/Concept |
|----|---------|-------|---------------------|
| 1.1 | Scalar/Vector/Tensor | Type classification | σ, v, p definitions |
| 1.2a | Orthogonality | a·b = 0 test | Dot product |
| 1.2b | Normal vectors | a×b perpendicular | Cross product |
| 1.2c | Tensor product | a⊗b matrix | Outer product |
| 1.2d | Angle between vectors | cos⁻¹(a·b/\|a\|\|b\|) | Dot product |
| 1.2e | Matrix multiplication | AB ≠ BA | Matrix ops |
| 1.2f | Determinant/trace | det, tr | Eigenvalue relations |
| 1.3 | Curl | ∇×v | Del operator |
| 1.4 | Diffusion vs convection | Definitions | Transport modes |
| 1.5 | Mass/molar | ρ = CM, ω = ρᵢ/ρ | Unit conversion |
| 1.6 | Fick's law | j = -D∇ρ | Diffusion flux |
| 1.7 | Continuity | ∂ρ/∂t + ∇·(ρv) = 0 | Mass balance |
| 2.1 | Membrane flux | j = DΦΔρ/L | Steady diffusion |
| 2.2 | Hindered diffusion | Dₕ = DK₁K₂ | Pore effects |
| 2.3 | Conv-diff simplification | Various limits | Equation reduction |
| 2.4 | 1D steady diffusion | D(d²ρ/dx²) + r = 0 | Governing equation |
| 2.5 | Partition coefficient | Concentration jumps | Hydrophobic effects |
| 2.6 | Unsteady diffusion | Bi, Fo, lumped | Microsphere release |
| 3.1 | Peclet number | Pe = Re×Sc | Dimensionless groups |
| 3.2 | Boundary layer | δc(x) | Mass transfer |
| 3.3 | Π-theorem | c²ρβ = const | Dimensional analysis |
| 3.4 | Similitude | Re matching | Model scaling |
| 3.5 | Darcy's law | q = -(κ/μ)∇p | Porous media |
| 4.1 | Definitions | Fluid, pressure, σᵢⱼ | Fundamentals |
| 4.2 | Cauchy stress | Eigenvalues, pressure | Stress analysis |
| 4.3 | Cylindrical ∇ | ∇·v derivation | Coordinate systems |
| 4.4 | Flow classification | Steady, incomp., irrot. | Property testing |
| 4.5 | Navier-Poisson | σ = -p + 2μD | Constitutive |
| 5.1 | Definitions | μ, Newtonian, no-slip | Fluid properties |
| 5.2 | Deformation rate | Dᵢⱼ derivation | Strain rate |
| 5.3 | Balance laws | Mass + momentum | Governing equations |
| 5.4 | Bioreactor stress | σ application | Rotating flow |
| 5.5 | Navier-Stokes | Full derivation | Core equation |
| 6.1 | Flow definitions | Steady, axisym., etc. | Classification |
| 6.2 | N-S practice | Derivation + statics | Solver setup |
| 6.3 | Cylindrical N-S | Pipe flow | Coordinate form |
| 6.4 | Bernoulli | p + ρgz + ρv²/2 = C | Inviscid flow |
| 6.5 | Viscoelasticity | SLS, Burgers | Material models |

---

## Appendix B: Equation Quick Reference

### Diffusion
```
Fick's Law:       j = -D∇ρ
Unsteady:         ∂ρ/∂t = D∇²ρ
Steady 1D:        d²ρ/dx² = 0  →  ρ(x) = Ax + B
```

### Convection-Diffusion
```
Full:             ∂ρ/∂t + v·∇ρ = D∇²ρ + r
Steady:           v·∇ρ = D∇²ρ + r
```

### Darcy Flow
```
Velocity:         v = -(κ/μ)∇p
With mass:        ∇·v = 0  →  ∇²p = 0 (Laplace)
```

### Navier-Stokes
```
Momentum:         ρ(∂v/∂t + v·∇v) = -∇p + μ∇²v + ρg
Mass:             ∇·v = 0 (incompressible)
```

### Constitutive Relations
```
Newtonian:        σᵢⱼ = -pδᵢⱼ + 2μDᵢⱼ
Power Law:        σᵢⱼ = k|γ̇|^(n-1) γ̇ᵢⱼ
```

### Dimensionless Numbers
```
Re = ρvL/μ        (inertia/viscous)
Sc = μ/(ρD)       (momentum/mass diffusion)
Pe = vL/D = Re×Sc (convection/diffusion)
Bi = hL/D         (external/internal)
Fo = Dt/L²        (dimensionless time)
```

### Analytical Solutions
```
Poiseuille:       vz(r) = (a²/4μ)(-dp/dz)(1 - r²/a²)
Couette:          vx(y) = (1/2μ)(dp/dx)(h²/4 - y²)
Taylor-Couette:   vθ(r) = Ar + B/r  (constants from BCs)
```

---

*Document generated for BioTransport Library development*
*Based on BMEN 341 Fall 2024 course materials*
*Texas A&M University Department of Biomedical Engineering*
