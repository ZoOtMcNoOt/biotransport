"""Machine-readable scientific contracts for native and Python numerical APIs.

The native registry describes compiled solver equations, numerical scope,
units, and the automated evidence that actually exists in this repository. A
separate Python registry describes governed adapter, reference, and workflow
modules together with their backend and retain/port/deprecate disposition.
Both are intentionally more conservative than marketing documentation: an
evidence record applies only to the claim it states, and numerical
verification is not biological validation.

Python-only reference solvers are never silently treated as native
implementations or allowed to inherit a compiled solver's evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final, Mapping, Optional, Union


class EvidenceLevel(str, Enum):
    """Strongest kind of automated evidence represented by one record.

    The ordering used by :func:`filter_contracts` is a convenience for
    discovery, not a claim that one test kind subsumes every weaker kind.
    None of these values denotes experimental or clinical validation.
    """

    UNTESTED = "untested"
    API = "api"
    BEHAVIOR = "behavior"
    INVARIANT = "invariant"
    ANALYTICAL = "analytical"
    CONVERGENCE = "convergence"


class PythonBackend(str, Enum):
    """Where a public Python numerical surface performs its substantive work."""

    NATIVE_ADAPTER = "native-adapter"
    MIXED = "mixed-python-native"
    PYTHON_REFERENCE = "python-reference"
    WORKFLOW = "python-workflow"


_EVIDENCE_RANK: Final[Mapping[EvidenceLevel, int]] = MappingProxyType(
    {
        EvidenceLevel.UNTESTED: 0,
        EvidenceLevel.API: 1,
        EvidenceLevel.BEHAVIOR: 2,
        EvidenceLevel.INVARIANT: 3,
        EvidenceLevel.ANALYTICAL: 4,
        EvidenceLevel.CONVERGENCE: 5,
    }
)


@dataclass(frozen=True)
class EvidenceRecord:
    """One scoped, reproducible statement about automated solver evidence.

    References use ``repository/path::test_or_function_name``.  The registry
    test checks both the path and selector, so deleted or renamed evidence
    cannot remain as an unnoticed claim.
    """

    level: EvidenceLevel
    claim: str
    references: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.level, EvidenceLevel):
            object.__setattr__(self, "level", EvidenceLevel(self.level))
        object.__setattr__(self, "references", tuple(self.references))
        if not self.claim.strip():
            raise ValueError("evidence claim must not be empty")
        if self.level is EvidenceLevel.UNTESTED:
            if self.references:
                raise ValueError("untested evidence cannot cite tests")
            return
        if not self.references:
            raise ValueError("tested evidence requires at least one exact reference")
        for reference in self.references:
            if "::" not in reference or reference.startswith("/") or "\\" in reference:
                raise ValueError(
                    "evidence references must use repository/path::selector"
                )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "level": self.level.value,
            "claim": self.claim,
            "references": list(self.references),
        }


@dataclass(frozen=True)
class SolverContract:
    """Immutable scientific and numerical contract for one native solver."""

    contract_id: str
    title: str
    native_symbols: tuple[str, ...]
    equation: str
    unknowns: tuple[str, ...]
    locations: tuple[str, ...]
    input_units: tuple[tuple[str, str], ...]
    output_units: tuple[tuple[str, str], ...]
    supported_dimensions: tuple[str, ...]
    supported_terms: tuple[str, ...]
    supported_boundary_conditions: tuple[str, ...]
    numerical_method: str
    stability_policy: str
    convergence_policy: str
    evidence: tuple[EvidenceRecord, ...]
    exclusions: tuple[str, ...]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        tuple_fields = (
            "native_symbols",
            "unknowns",
            "locations",
            "input_units",
            "output_units",
            "supported_dimensions",
            "supported_terms",
            "supported_boundary_conditions",
            "evidence",
            "exclusions",
            "warnings",
        )
        for field_name in tuple_fields:
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))
        object.__setattr__(
            self,
            "input_units",
            tuple((name, unit) for name, unit in self.input_units),
        )
        object.__setattr__(
            self,
            "output_units",
            tuple((name, unit) for name, unit in self.output_units),
        )

        required_text = (
            self.contract_id,
            self.title,
            self.equation,
            self.numerical_method,
            self.stability_policy,
            self.convergence_policy,
        )
        if any(not value.strip() for value in required_text):
            raise ValueError("contract text fields must not be empty")
        if not self.native_symbols:
            raise ValueError("a solver contract requires a native symbol")
        if not self.unknowns or not self.locations:
            raise ValueError("unknowns and storage locations must be explicit")
        if not self.input_units or not self.output_units:
            raise ValueError("input and output units must be explicit")
        if not self.supported_dimensions or not self.supported_terms:
            raise ValueError("dimensions and supported terms must be explicit")
        if not self.evidence:
            raise ValueError("evidence must be recorded, including untested status")
        for unit_table in (self.input_units, self.output_units):
            names = [name for name, _unit in unit_table]
            if len(names) != len(set(names)):
                raise ValueError("unit quantity names must be unique")

    @property
    def evidence_level(self) -> EvidenceLevel:
        """Return the strongest registered automated evidence kind."""

        return max(self.evidence, key=lambda item: _EVIDENCE_RANK[item.level]).level

    def unit_for(self, quantity: str, *, output: bool = False) -> str:
        """Look up a documented unit by exact quantity name."""

        table = self.output_units if output else self.input_units
        for name, unit in table:
            if name == quantity:
                return unit
        direction = "output" if output else "input"
        raise KeyError(f"{quantity!r} is not a documented {direction} quantity")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "contract_id": self.contract_id,
            "title": self.title,
            "native_symbols": list(self.native_symbols),
            "equation": self.equation,
            "unknowns": list(self.unknowns),
            "locations": list(self.locations),
            "input_units": dict(self.input_units),
            "output_units": dict(self.output_units),
            "supported_dimensions": list(self.supported_dimensions),
            "supported_terms": list(self.supported_terms),
            "supported_boundary_conditions": list(self.supported_boundary_conditions),
            "numerical_method": self.numerical_method,
            "stability_policy": self.stability_policy,
            "convergence_policy": self.convergence_policy,
            "evidence_level": self.evidence_level.value,
            "evidence": [record.to_dict() for record in self.evidence],
            "exclusions": list(self.exclusions),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class PythonNumericalContract:
    """Immutable contract for one public Python numerical/workflow module.

    These records are deliberately separate from :class:`SolverContract`.
    They disclose Python orchestration and reference implementations without
    pretending that they are compiled native solver entry points or inherit a
    native solver's verification evidence.
    """

    contract_id: str
    title: str
    module: str
    public_symbols: tuple[str, ...]
    category: str
    backend: PythonBackend
    mathematical_scope: str
    numerical_method: str
    failure_policy: str
    evidence: tuple[EvidenceRecord, ...]
    disposition: str
    exclusions: tuple[str, ...]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in ("public_symbols", "evidence", "exclusions", "warnings"):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))
        if not isinstance(self.backend, PythonBackend):
            object.__setattr__(self, "backend", PythonBackend(self.backend))
        required_text = (
            self.contract_id,
            self.title,
            self.module,
            self.category,
            self.mathematical_scope,
            self.numerical_method,
            self.failure_policy,
            self.disposition,
        )
        if any(not value.strip() for value in required_text):
            raise ValueError("Python numerical contract text fields must not be empty")
        if not self.module.startswith("biotransport."):
            raise ValueError(
                "Python numerical contract module must be package-qualified"
            )
        if not self.public_symbols:
            raise ValueError("Python numerical contract requires public symbols")
        if len(self.public_symbols) != len(set(self.public_symbols)):
            raise ValueError("Python numerical contract symbols must be unique")
        if not self.evidence:
            raise ValueError("Python numerical contract requires an evidence record")

    @property
    def evidence_level(self) -> EvidenceLevel:
        """Return the strongest scoped evidence represented by the record."""

        return max(self.evidence, key=lambda item: _EVIDENCE_RANK[item.level]).level

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-serializable representation."""

        return {
            "contract_id": self.contract_id,
            "title": self.title,
            "module": self.module,
            "public_symbols": list(self.public_symbols),
            "category": self.category,
            "backend": self.backend.value,
            "mathematical_scope": self.mathematical_scope,
            "numerical_method": self.numerical_method,
            "failure_policy": self.failure_policy,
            "evidence_level": self.evidence_level.value,
            "evidence": [record.to_dict() for record in self.evidence],
            "disposition": self.disposition,
            "exclusions": list(self.exclusions),
            "warnings": list(self.warnings),
        }


EVIDENCE_DISCLAIMER: Final[str] = (
    "Evidence levels summarize repository tests for scoped numerical claims. "
    "They are not experimental or biological validation, an uncertainty "
    "quantification, or a claim of ASME V&V 20 compliance."
)


_SCALAR_INPUT_UNITS = (
    ("field", "user-selected field unit"),
    ("diffusivity", "length^2/time"),
    ("coordinate", "length"),
    ("time", "time"),
)
_SCALAR_OUTPUT_UNITS = (
    ("field", "same as input field"),
    ("time", "time"),
)
_SCALAR_BOUNDARIES = (
    "Dirichlet field value",
    "Neumann outward-normal field derivative",
)
_LEGACY_SCALAR_BOUNDARIES = _SCALAR_BOUNDARIES + ("Robin a*u + b*du/dn = c",)
_API_EXPORT_EVIDENCE = EvidenceRecord(
    EvidenceLevel.API,
    "The compiled symbol is exported through an explicit public namespace; "
    "this does not exercise its numerical update.",
    (
        "python/tests/test_public_namespaces.py::"
        "test_namespace_exports_are_explicit_and_discoverable",
    ),
)


_CONTRACTS: Final[tuple[SolverContract, ...]] = (
    SolverContract(
        contract_id="transport.canonical_explicit",
        title="Canonical conservative scalar transport",
        native_symbols=("solve_transport",),
        equation=("dc/dt = div(D grad(c)) - div(v c) + R(c,x,y,t)"),
        unknowns=("scalar field c",),
        locations=("vertex-centred control volumes on StructuredMesh",),
        input_units=(
            ("concentration", "user-selected concentration unit"),
            ("diffusivity", "length^2/time"),
            ("velocity", "length/time"),
            ("reaction_rate", "concentration/time"),
            ("coordinate", "length"),
            ("time", "time"),
        ),
        output_units=(
            ("concentration", "same as input concentration"),
            ("time", "time"),
            ("mass_diagnostic", "concentration*length^dimension"),
        ),
        supported_dimensions=("1D", "2D"),
        supported_terms=(
            "scalar or nodal isotropic diffusion",
            "uniform or nodal prescribed velocity",
            "composable local reactions and sources",
            "conservative first-order upwind advection",
        ),
        supported_boundary_conditions=(
            "Dirichlet concentration",
            "Neumann outward-normal concentration derivative",
            "Robin a*c + b*dc/dn = value",
        ),
        numerical_method=(
            "Conservative nodal finite volume with harmonic diffusive face "
            "coefficients, upwind advective face values, and Forward Euler."
        ),
        stability_policy=(
            "A supplied step is rejected above the explicit transport bound. "
            "Automatic stepping also needs a declared reaction derivative bound; "
            "otherwise the user must provide the step. The last step is shortened "
            "to land exactly on final_time."
        ),
        convergence_policy=(
            "A smooth diffusion eigenmode measures second-order spatial convergence "
            "when temporal error is suppressed. A heterogeneous diffusion/advection "
            "manufactured case measures the expected first-order upwind behavior, and "
            "one composed reaction case measures first-order temporal convergence. "
            "No blanket order claim is made outside those configurations."
        ),
        evidence=(
            EvidenceRecord(
                EvidenceLevel.CONVERGENCE,
                "Independent continuum/manufactured references measure second-order "
                "diffusion space, first-order heterogeneous upwind space, and first-order "
                "reaction time while separate cases exercise conservation and exact time.",
                (
                    "cpp/tests/physics/test_transport_solver_science.cpp::"
                    "canonicalSolveHasSecondOrderSpatialGridConvergence",
                    "cpp/tests/physics/test_transport_solver_science.cpp::"
                    "heterogeneousCoefficientsAndMixedBoundariesConverge",
                    "cpp/tests/physics/test_transport_solver_science.cpp::"
                    "reactionsComposeAndConvergeInTime",
                    "cpp/tests/physics/test_transport_solver_science.cpp::"
                    "manufacturedQuadraticIsStationaryIn2D",
                    "cpp/tests/physics/test_transport_solver_science.cpp::"
                    "conservativeFluxesPreserveMassWithVariableFields",
                    "cpp/tests/physics/test_transport_solver_science.cpp::"
                    "finalTimeIsExactAndLastStepIsShortened",
                ),
            ),
        ),
        exclusions=(
            "anisotropic or tensor diffusion",
            "nonlocal reactions",
            "CENTRAL, HYBRID, and QUICK advection on this canonical path",
            "3D meshes",
        ),
        warnings=(
            "Neumann data is a field derivative, not a Fickian flux.",
            "A numerically verified equation is not a validated biological model.",
        ),
    ),
    SolverContract(
        contract_id="transport.legacy_explicit_fd",
        title="Legacy ExplicitFD transport facade",
        native_symbols=("ExplicitFD",),
        equation="Selected legacy diffusion/advection/reaction equation from TransportProblem",
        unknowns=("scalar field",),
        locations=("StructuredMesh nodes",),
        input_units=_SCALAR_INPUT_UNITS
        + (
            ("velocity", "length/time"),
            ("reaction_rate", "field/time"),
        ),
        output_units=_SCALAR_OUTPUT_UNITS
        + (("mass_statistic", "field*length^dimension"),),
        supported_dimensions=("1D", "2D"),
        supported_terms=(
            "constant or nodal diffusion",
            "prescribed advection",
            "one local reaction callback or built-in reaction",
        ),
        supported_boundary_conditions=_LEGACY_SCALAR_BOUNDARIES,
        numerical_method=(
            "Facade selecting legacy Forward-Euler diffusion, reaction-diffusion, "
            "variable-diffusion, or advection-diffusion implementations."
        ),
        stability_policy=(
            "Chooses a diffusion/advection step from a safety factor, but generic "
            "reaction callbacks have no complete reaction stability certificate."
        ),
        convergence_policy=(
            "Individual uniform-source and reaction reductions have regression "
            "comparisons; no unified convergence claim exists for the facade."
        ),
        evidence=(
            EvidenceRecord(
                EvidenceLevel.ANALYTICAL,
                "Facade paths are compared with uniform constant-source and kinetic "
                "ODE reductions and tested for exact requested end time.",
                (
                    "cpp/tests/physics/test_explicit_fd_run.cpp::"
                    "testExplicitFDRunUsesStableDtAndPinsDirichlet",
                    "cpp/tests/physics/test_explicit_fd_run_constant_source.cpp::"
                    "testExplicitFDConstantSource1DUniformGrowth",
                    "cpp/tests/physics/test_explicit_fd_run_logistic.cpp::"
                    "testExplicitFDLogistic1DUniformGrowth",
                    "cpp/tests/physics/test_explicit_fd_run_michaelis_menten.cpp::"
                    "testExplicitFDMichaelisMenten1DUniformDecay",
                ),
            ),
        ),
        exclusions=(
            "complete a priori stability certification for arbitrary reactions",
            "the canonical diagnostic detail returned by solve_transport",
        ),
        warnings=(
            "This is a compatibility facade; prefer solve_transport for new work.",
            "Generic callback reactions are checked at each candidate update, not "
            "certified before the run begins.",
        ),
    ),
    SolverContract(
        contract_id="diffusion.forward_euler_1d_2d",
        title="Legacy explicit diffusion",
        native_symbols=("DiffusionSolver",),
        equation="du/dt = D laplacian(u)",
        unknowns=("scalar field u",),
        locations=("StructuredMesh nodes",),
        input_units=_SCALAR_INPUT_UNITS,
        output_units=_SCALAR_OUTPUT_UNITS,
        supported_dimensions=("1D", "2D"),
        supported_terms=("constant isotropic diffusion",),
        supported_boundary_conditions=_LEGACY_SCALAR_BOUNDARIES,
        numerical_method="Centred finite differences with Forward Euler time stepping.",
        stability_policy=(
            "The caller supplies dt; solve rejects steps above the standard explicit "
            "constant-diffusion limit."
        ),
        convergence_policy=(
            "Only qualitative diffusion behavior is directly tested for this class; "
            "no measured order is registered."
        ),
        evidence=(
            EvidenceRecord(
                EvidenceLevel.BEHAVIOR,
                "A Gaussian field broadens with a reduced peak under stable stepping.",
                ("python/tests/test_diffusion.py::test_diffusion_solver",),
            ),
        ),
        exclusions=(
            "variable diffusivity",
            "advection",
            "reaction",
        ),
        warnings=(
            "Default boundaries are zero Dirichlet.",
            "The legacy 1D/2D boundary update is not the canonical finite-volume path.",
        ),
    ),
    SolverContract(
        contract_id="diffusion.forward_euler_3d",
        title="Conservative explicit 3D diffusion",
        native_symbols=("DiffusionSolver3D",),
        equation="du/dt = D laplacian(u)",
        unknowns=("scalar field u",),
        locations=("vertex-centred control volumes on StructuredMesh3D",),
        input_units=_SCALAR_INPUT_UNITS,
        output_units=_SCALAR_OUTPUT_UNITS,
        supported_dimensions=("3D",),
        supported_terms=("constant isotropic diffusion",),
        supported_boundary_conditions=_SCALAR_BOUNDARIES,
        numerical_method="Conservative vertex-centred finite volume with Forward Euler.",
        stability_policy=(
            "dt must not exceed 1/[2D(1/dx^2+1/dy^2+1/dz^2)]; rejected steps "
            "do not mutate state."
        ),
        convergence_policy=(
            "Linear steady states and conservation are tested, but automated spatial "
            "or temporal order measurement is not registered."
        ),
        evidence=(
            EvidenceRecord(
                EvidenceLevel.ANALYTICAL,
                "A three-dimensional linear Neumann solution remains stationary and "
                "closed-boundary control-volume mass is conserved.",
                (
                    "cpp/tests/physics/test_specialized_diffusion_science.cpp::"
                    "explicit3DPreservesLinearNeumannSolution",
                    "cpp/tests/physics/test_specialized_diffusion_science.cpp::"
                    "explicit3DConservesControlVolumeMass",
                ),
            ),
        ),
        exclusions=("variable diffusivity", "reaction", "Robin boundaries"),
        warnings=("Neumann data is du/dn, not diffusive flux.",),
    ),
    SolverContract(
        contract_id="diffusion.nonuniform_forward_euler_1d",
        title="Conservative nonuniform 1D diffusion",
        native_symbols=("NonuniformDiffusion1D",),
        equation="dc/dt = d/dx(D(x) dc/dx)",
        unknowns=("non-negative scalar concentration c",),
        locations=("nodes of a fixed fitted NonuniformMesh1D control-volume mesh",),
        input_units=(
            ("concentration", "user-selected concentration unit"),
            ("node_coordinate", "length"),
            ("nodal_diffusivity", "length^2/time"),
            ("outward_concentration_derivative", "concentration/length"),
            ("time", "time"),
        ),
        output_units=(
            ("concentration", "same as input concentration"),
            ("face_fickian_flux", "concentration*length/time"),
            ("integrated_concentration", "concentration*length"),
            ("mass_balance_error", "concentration*length"),
            ("time", "time"),
        ),
        supported_dimensions=("1D",),
        supported_terms=(
            "scalar or nodal non-negative diffusivity",
            "harmonic face diffusivity across material discontinuities",
        ),
        supported_boundary_conditions=(
            "Dirichlet concentration",
            "Neumann outward-normal concentration derivative",
        ),
        numerical_method=(
            "Conservative node-centred finite volume on a fitted nonuniform mesh, "
            "using one harmonic-mean Fickian face flux and Forward Euler time stepping."
        ),
        stability_policy=(
            "max_stable_time_step returns the exact local Forward Euler monotonicity "
            "bound min_i[V_i/sum(face conductances)] over non-Dirichlet nodes; an "
            "unstable or non-finite step is rejected before state mutation. solve_until "
            "uses stable equal substeps and lands at the requested absolute final time."
        ),
        convergence_policy=(
            "Second-order spatial convergence is measured for a smooth sine mode on "
            "smoothly stretched meshes while time error is suppressed. This does not "
            "establish second order at discontinuities, for nonsmooth meshes, or in time."
        ),
        evidence=(
            EvidenceRecord(
                EvidenceLevel.CONVERGENCE,
                "A smooth manufactured mode on 20, 40, and 80-cell stretched meshes "
                "has successive L2 error ratios greater than 3.2; the Python binding "
                "independently exercises the same refinement claim.",
                (
                    "cpp/tests/core/test_nonuniform_diffusion_science.cpp::"
                    "manufacturedSolutionConvergesOnSmoothlyStretchedMeshes",
                    "python/tests/test_nonuniform_diffusion.py::"
                    "test_manufactured_solution_is_second_order_on_stretched_meshes",
                ),
            ),
            EvidenceRecord(
                EvidenceLevel.INVARIANT,
                "Uniform-grid stencil parity, one steady flux across discontinuous "
                "diffusivity, closed-domain integrated conservation, outward-normal "
                "boundary signs, and atomic rejection above the stability limit are "
                "directly checked.",
                (
                    "cpp/tests/core/test_nonuniform_diffusion_science.cpp::"
                    "uniformGridReducesToStandardFiniteVolumeStencil",
                    "cpp/tests/core/test_nonuniform_diffusion_science.cpp::"
                    "discontinuousDiffusivityMaintainsOneConservativeFaceFlux",
                    "cpp/tests/core/test_nonuniform_diffusion_science.cpp::"
                    "closedIrregularMeshConservesIntegratedMass",
                    "cpp/tests/core/test_nonuniform_diffusion_science.cpp::"
                    "outwardNormalBoundarySignsMatchFicksLaw",
                    "cpp/tests/core/test_nonuniform_diffusion_science.cpp::"
                    "invalidInputsAndUnstableStepsFailLoudly",
                    "python/tests/test_nonuniform_diffusion.py::"
                    "test_invalid_material_boundary_and_unstable_steps_fail_loudly",
                ),
            ),
        ),
        exclusions=(
            "higher-dimensional or unstructured meshes",
            "adaptive refinement, moving meshes, or remeshing",
            "cylindrical or spherical metric factors",
            "advection or reaction",
            "nonlinear or tensor diffusivity",
            "contact resistance",
            "Robin boundaries",
        ),
        warnings=(
            "Neumann data is dc/dn; physical outward Fickian flux is -D dc/dn, so a "
            "positive derivative adds integrated concentration.",
            "Default boundaries are closed zero-Neumann walls.",
            "Strong mesh refinement or large local conductance can make the explicit "
            "stability limit prohibitively small.",
            "Integrated concentration assumes unit cross-sectional area unless the "
            "caller supplies a separate geometric interpretation.",
        ),
    ),
    SolverContract(
        contract_id="diffusion.crank_nicolson",
        title="Crank-Nicolson diffusion",
        native_symbols=("CrankNicolsonDiffusion",),
        equation="du/dt = D laplacian(u)",
        unknowns=("scalar field u",),
        locations=("StructuredMesh nodes",),
        input_units=_SCALAR_INPUT_UNITS,
        output_units=_SCALAR_OUTPUT_UNITS,
        supported_dimensions=("1D", "2D"),
        supported_terms=("constant isotropic diffusion",),
        supported_boundary_conditions=_SCALAR_BOUNDARIES,
        numerical_method="Crank-Nicolson with iterative linear solves.",
        stability_policy=(
            "A-stable for linear diffusion, but not L-stable or positivity preserving; "
            "linear non-convergence fails atomically."
        ),
        convergence_policy=(
            "Second-order temporal convergence is measured for a smooth 1D mode; that "
            "claim is not extended to nonsmooth data or time-dependent boundaries."
        ),
        evidence=(
            EvidenceRecord(
                EvidenceLevel.CONVERGENCE,
                "A smooth one-dimensional mode demonstrates second-order temporal "
                "convergence; conservation and non-convergence behavior are separate checks.",
                (
                    "python/tests/test_crank_nicolson.py::test_temporal_accuracy_1d",
                    "cpp/tests/physics/test_specialized_diffusion_science.cpp::"
                    "crankNicolsonConservesControlVolumeMass",
                    "cpp/tests/physics/test_specialized_diffusion_science.cpp::"
                    "crankNicolsonFailureDoesNotAdvanceState",
                ),
            ),
        ),
        exclusions=(
            "variable diffusivity",
            "reaction",
            "3D meshes",
            "Robin boundaries",
        ),
        warnings=(
            "Large stable steps can oscillate or lose positivity.",
            "Neumann data is du/dn, not diffusive flux.",
        ),
    ),
    SolverContract(
        contract_id="diffusion.adi_2d",
        title="Symmetric ADI diffusion in 2D",
        native_symbols=("ADIDiffusion2D",),
        equation="du/dt = D (d2u/dx2 + d2u/dy2)",
        unknowns=("scalar field u",),
        locations=("StructuredMesh nodes",),
        input_units=_SCALAR_INPUT_UNITS,
        output_units=_SCALAR_OUTPUT_UNITS,
        supported_dimensions=("2D",),
        supported_terms=("constant isotropic diffusion",),
        supported_boundary_conditions=_SCALAR_BOUNDARIES,
        numerical_method="Symmetric x/2-y-x/2 alternating-direction implicit split.",
        stability_policy=(
            "Directional diffusion solves are unconditionally stable, but accuracy and "
            "positivity still constrain useful steps."
        ),
        convergence_policy=(
            "Second-order temporal convergence is measured for a smooth manufactured "
            "case with time-independent boundary data."
        ),
        evidence=(
            EvidenceRecord(
                EvidenceLevel.CONVERGENCE,
                "The two-dimensional symmetric split demonstrates second-order temporal "
                "convergence and preserves closed-boundary mass and linear Neumann data.",
                (
                    "cpp/tests/physics/test_specialized_diffusion_science.cpp::"
                    "adiIsSecondOrderInTime",
                    "cpp/tests/physics/test_specialized_diffusion_science.cpp::"
                    "adi2DConservesMassAndPreservesLinearNeumannData",
                ),
            ),
        ),
        exclusions=("variable diffusivity", "reaction", "Robin boundaries"),
        warnings=(
            "Unconditional stability is not unconditional accuracy or positivity.",
        ),
    ),
    SolverContract(
        contract_id="diffusion.adi_3d",
        title="Symmetric ADI diffusion in 3D",
        native_symbols=("ADIDiffusion3D",),
        equation="du/dt = D (d2u/dx2 + d2u/dy2 + d2u/dz2)",
        unknowns=("scalar field u",),
        locations=("StructuredMesh3D nodes",),
        input_units=_SCALAR_INPUT_UNITS,
        output_units=_SCALAR_OUTPUT_UNITS,
        supported_dimensions=("3D",),
        supported_terms=("constant isotropic diffusion",),
        supported_boundary_conditions=_SCALAR_BOUNDARIES,
        numerical_method="Symmetric x/2-y/2-z-y/2-x/2 ADI split.",
        stability_policy=(
            "Directional diffusion solves are unconditionally stable; useful steps "
            "remain limited by accuracy and possible positivity loss."
        ),
        convergence_policy=(
            "Conservation and a linear steady state are tested. No automated 3D order "
            "measurement is registered."
        ),
        evidence=(
            EvidenceRecord(
                EvidenceLevel.ANALYTICAL,
                "A three-dimensional linear Neumann field remains stationary and "
                "closed-boundary mass is conserved.",
                (
                    "cpp/tests/physics/test_specialized_diffusion_science.cpp::"
                    "adi3DConservesMassAndPreservesLinearNeumannData",
                ),
            ),
        ),
        exclusions=("variable diffusivity", "reaction", "Robin boundaries"),
        warnings=("No 3D temporal-order claim is registered.",),
    ),
    SolverContract(
        contract_id="diffusion.backward_euler_2d",
        title="Sparse Backward-Euler diffusion in 2D",
        native_symbols=("ImplicitDiffusion2D",),
        equation="du/dt = div(D grad(u)) + source(x,y,t)",
        unknowns=("scalar field u",),
        locations=("vertex-centred control volumes on StructuredMesh",),
        input_units=_SCALAR_INPUT_UNITS + (("source", "field/time"),),
        output_units=_SCALAR_OUTPUT_UNITS
        + (("algebraic_residual", "solver-normalized residual"),),
        supported_dimensions=("2D",),
        supported_terms=(
            "scalar or nodal isotropic diffusion",
            "prescribed local source independent of u",
        ),
        supported_boundary_conditions=_SCALAR_BOUNDARIES,
        numerical_method=(
            "Conservative Backward Euler with harmonic face diffusivity and an "
            "Eigen sparse linear solve."
        ),
        stability_policy=(
            "L-stable for linear diffusion; algebraic failure is reported. Large steps "
            "can still be inaccurate."
        ),
        convergence_policy=(
            "Discrete variable-coefficient equilibria and source balances are tested; "
            "no automated order measurement is registered."
        ),
        evidence=(
            EvidenceRecord(
                EvidenceLevel.ANALYTICAL,
                "A discrete variable-coefficient flux equilibrium is preserved and a "
                "source-plus-Neumann mass balance closes.",
                (
                    "cpp/tests/physics/test_specialized_diffusion_science.cpp::"
                    "implicit2DPreservesDiscreteVariableCoefficientFlux",
                    "cpp/tests/physics/test_specialized_diffusion_science.cpp::"
                    "implicit2DSourceAndNeumannMassBalance",
                ),
            ),
        ),
        exclusions=("advection", "state-dependent reaction", "Robin boundaries"),
        warnings=("Availability requires the Eigen sparse backend.",),
    ),
    SolverContract(
        contract_id="diffusion.backward_euler_3d",
        title="Sparse Backward-Euler diffusion in 3D",
        native_symbols=("ImplicitDiffusion3D",),
        equation="du/dt = div(D grad(u)) + source(x,y,z,t)",
        unknowns=("scalar field u",),
        locations=("vertex-centred control volumes on StructuredMesh3D",),
        input_units=_SCALAR_INPUT_UNITS + (("source", "field/time"),),
        output_units=_SCALAR_OUTPUT_UNITS
        + (("algebraic_residual", "solver-normalized residual"),),
        supported_dimensions=("3D",),
        supported_terms=(
            "scalar or nodal isotropic diffusion",
            "prescribed local source independent of u",
        ),
        supported_boundary_conditions=_SCALAR_BOUNDARIES,
        numerical_method=(
            "Conservative Backward Euler with harmonic face diffusivity and an "
            "Eigen sparse linear solve."
        ),
        stability_policy="L-stable for linear diffusion; algebraic failure is reported.",
        convergence_policy=(
            "A linear Neumann steady state is tested; no automated 3D order "
            "measurement is registered."
        ),
        evidence=(
            EvidenceRecord(
                EvidenceLevel.ANALYTICAL,
                "A three-dimensional linear Neumann field is preserved by a converged "
                "implicit step.",
                (
                    "cpp/tests/physics/test_specialized_diffusion_science.cpp::"
                    "implicit3DPreservesLinearNeumannSolution",
                ),
            ),
        ),
        exclusions=("advection", "state-dependent reaction", "Robin boundaries"),
        warnings=("Availability requires the Eigen sparse backend.",),
    ),
    SolverContract(
        contract_id="transport.legacy_advection_diffusion",
        title="Legacy advection-diffusion",
        native_symbols=("AdvectionDiffusionSolver",),
        equation="du/dt + v dot grad(u) = D laplacian(u)",
        unknowns=("scalar field u",),
        locations=("StructuredMesh nodes",),
        input_units=_SCALAR_INPUT_UNITS + (("velocity", "length/time"),),
        output_units=_SCALAR_OUTPUT_UNITS,
        supported_dimensions=("1D", "2D"),
        supported_terms=(
            "constant isotropic diffusion",
            "uniform or nodal prescribed velocity",
            "upwind, central, or Peclet-switched hybrid advection",
        ),
        supported_boundary_conditions=_LEGACY_SCALAR_BOUNDARIES,
        numerical_method=(
            "Forward Euler with centred diffusion and legacy pointwise upwind, "
            "central, or Peclet-switched advection."
        ),
        stability_policy=(
            "max_time_step combines legacy diffusion and directional advection bounds; "
            "central differencing is reported stable only below its cell-Peclet criterion."
        ),
        convergence_policy=(
            "Qualitative pulse transport and diffusion-dominated behavior are tested; "
            "no measured order is registered."
        ),
        evidence=(
            EvidenceRecord(
                EvidenceLevel.BEHAVIOR,
                "A pulse moves in the prescribed direction and a diffusion-dominated "
                "case smooths under stable stepping.",
                (
                    "python/tests/test_advection_diffusion.py::test_pure_advection",
                    "python/tests/test_advection_diffusion.py::test_diffusion_dominated",
                ),
            ),
        ),
        exclusions=("reaction", "conservative velocity divergence"),
        warnings=(
            "QUICK is unsupported until a genuine verified third-order stencil exists.",
            "Prefer solve_transport when conservative advection is required.",
        ),
    ),
    SolverContract(
        contract_id="reaction.generic_explicit",
        title="Generic explicit reaction-diffusion",
        native_symbols=("ReactionDiffusionSolver",),
        equation="du/dt = D laplacian(u) + R(u,x,y,t)",
        unknowns=("scalar field u",),
        locations=("StructuredMesh nodes",),
        input_units=_SCALAR_INPUT_UNITS + (("reaction_rate", "field/time"),),
        output_units=_SCALAR_OUTPUT_UNITS,
        supported_dimensions=("1D", "2D"),
        supported_terms=("constant diffusion", "arbitrary local reaction callback"),
        supported_boundary_conditions=_LEGACY_SCALAR_BOUNDARIES,
        numerical_method="Centred diffusion and local reaction with Forward Euler.",
        stability_policy=(
            "The explicit diffusion limit is enforced. No universal a priori callback "
            "reaction bound exists, so every complete diffusion/reaction candidate is "
            "checked for finiteness and, by default, nonnegativity before state mutation."
        ),
        convergence_policy=(
            "Uniform logistic and Michaelis-Menten reductions demonstrate first-order "
            "temporal convergence for those specific callbacks."
        ),
        evidence=(
            EvidenceRecord(
                EvidenceLevel.CONVERGENCE,
                "Uniform logistic and Michaelis-Menten PDE reductions agree with their "
                "analytical ODEs and exhibit first-order temporal convergence.",
                (
                    "cpp/tests/physics/test_logistic_reaction_diffusion.cpp::"
                    "testUniformFieldMatchesLogisticOde",
                    "cpp/tests/physics/test_michaelis_menten_reaction_diffusion.cpp::"
                    "testUniformFieldMatchesMichaelisMentenOde",
                ),
            ),
            EvidenceRecord(
                EvidenceLevel.BEHAVIOR,
                "Non-finite callbacks and negative concentration candidates fail "
                "transactionally; signed-scalar behavior requires an explicit opt-out.",
                (
                    "cpp/tests/physics/test_legacy_reaction_safety.cpp::"
                    "genericCallbackFailuresAreFiniteAndTransactional",
                ),
            ),
        ),
        exclusions=("variable diffusivity", "advection"),
        warnings=(
            "An arbitrary callback has no automatic a priori reaction stability certificate.",
            "Unsafe concentration candidates raise instead of being clipped or committed.",
        ),
    ),
    SolverContract(
        contract_id="reaction.linear_imex_1d_2d",
        title="Linear-decay reaction-diffusion",
        native_symbols=("LinearReactionDiffusionSolver",),
        equation="du/dt = D laplacian(u) - k*u",
        unknowns=("scalar field u",),
        locations=("StructuredMesh nodes",),
        input_units=_SCALAR_INPUT_UNITS + (("decay_rate", "1/time"),),
        output_units=_SCALAR_OUTPUT_UNITS,
        supported_dimensions=("1D", "2D"),
        supported_terms=("constant diffusion", "first-order decay"),
        supported_boundary_conditions=_LEGACY_SCALAR_BOUNDARIES,
        numerical_method="Forward-Euler diffusion with Backward-Euler linear decay.",
        stability_policy=(
            "Decay is unconditionally stable and positivity preserving in isolation; "
            "the explicit diffusion limit still applies."
        ),
        convergence_policy="Measured first-order temporal convergence for uniform decay.",
        evidence=(
            EvidenceRecord(
                EvidenceLevel.CONVERGENCE,
                "A uniform field agrees with exponential decay within tolerance and "
                "demonstrates first-order Backward-Euler convergence.",
                (
                    "cpp/tests/physics/test_linear_reaction_diffusion.cpp::"
                    "testUniformFieldMatchesExponentialDecay",
                ),
            ),
        ),
        exclusions=("nonlinear decay", "variable diffusivity"),
        warnings=("The combined IMEX method is first order in time.",),
    ),
    SolverContract(
        contract_id="reaction.linear_imex_3d",
        title="Linear-decay reaction-diffusion in 3D",
        native_symbols=("LinearReactionDiffusionSolver3D",),
        equation="du/dt = D laplacian(u) - k*u",
        unknowns=("scalar field u",),
        locations=("vertex-centred control volumes on StructuredMesh3D",),
        input_units=_SCALAR_INPUT_UNITS + (("decay_rate", "1/time"),),
        output_units=_SCALAR_OUTPUT_UNITS,
        supported_dimensions=("3D",),
        supported_terms=("constant diffusion", "first-order decay"),
        supported_boundary_conditions=_SCALAR_BOUNDARIES,
        numerical_method="Forward-Euler diffusion with Backward-Euler linear decay.",
        stability_policy="The explicit 3D diffusion limit applies; decay is implicit.",
        convergence_policy=(
            "The exact discrete Backward-Euler update is tested. No measured continuum "
            "convergence order is registered for this 3D class."
        ),
        evidence=(
            EvidenceRecord(
                EvidenceLevel.BEHAVIOR,
                "A uniform field matches the exact repeated Backward-Euler decay update.",
                (
                    "cpp/tests/physics/test_specialized_diffusion_science.cpp::"
                    "explicit3DReactionContractsAreHonest",
                ),
            ),
        ),
        exclusions=("nonlinear decay", "variable diffusivity", "Robin boundaries"),
        warnings=("The combined IMEX method is first order in time.",),
    ),
    SolverContract(
        contract_id="reaction.logistic_specialized",
        title="Specialized logistic reaction-diffusion",
        native_symbols=("LogisticReactionDiffusionSolver",),
        equation="du/dt = D laplacian(u) + r*u*(1-u/K)",
        unknowns=("scalar field u",),
        locations=("StructuredMesh nodes",),
        input_units=_SCALAR_INPUT_UNITS
        + (("growth_rate", "1/time"), ("carrying_capacity", "field")),
        output_units=_SCALAR_OUTPUT_UNITS,
        supported_dimensions=("1D", "2D"),
        supported_terms=("constant diffusion", "logistic reaction"),
        supported_boundary_conditions=_LEGACY_SCALAR_BOUNDARIES,
        numerical_method="Centred diffusion and logistic reaction with Forward Euler.",
        stability_policy=(
            "The diffusion bound is enforced. The reaction has no universal a priori "
            "bound, so complete candidates are checked for finiteness and nonnegativity "
            "before mutation."
        ),
        convergence_policy=(
            "No direct numerical test or measured order is registered for this specialized "
            "class. Generic-callback tests do not automatically verify this implementation."
        ),
        evidence=(
            _API_EXPORT_EVIDENCE,
            EvidenceRecord(
                EvidenceLevel.BEHAVIOR,
                "A reaction-unstable negative candidate is rejected without mutating state.",
                (
                    "cpp/tests/physics/test_legacy_reaction_safety.cpp::"
                    "specializedWrappersEnforceReactionAwarePositivity",
                ),
            ),
        ),
        exclusions=("variable diffusivity", "advection"),
        warnings=(
            "No measured convergence order is registered for this specialization.",
        ),
    ),
    SolverContract(
        contract_id="reaction.michaelis_menten_specialized",
        title="Specialized Michaelis-Menten reaction-diffusion",
        native_symbols=("MichaelisMentenReactionDiffusionSolver",),
        equation="du/dt = D laplacian(u) - Vmax*u/(Km+u)",
        unknowns=("scalar field u",),
        locations=("StructuredMesh nodes",),
        input_units=_SCALAR_INPUT_UNITS + (("vmax", "field/time"), ("km", "field")),
        output_units=_SCALAR_OUTPUT_UNITS,
        supported_dimensions=("1D", "2D"),
        supported_terms=("constant diffusion", "Michaelis-Menten consumption"),
        supported_boundary_conditions=_LEGACY_SCALAR_BOUNDARIES,
        numerical_method=(
            "Centred diffusion and explicit Michaelis-Menten consumption with Forward Euler."
        ),
        stability_policy=(
            "The diffusion bound is enforced; Km and Vmax are validated, the singular "
            "reaction denominator is rejected, and complete candidates must remain finite "
            "and nonnegative before mutation."
        ),
        convergence_policy=(
            "No direct numerical test or measured order is registered for this specialized "
            "class."
        ),
        evidence=(
            _API_EXPORT_EVIDENCE,
            EvidenceRecord(
                EvidenceLevel.BEHAVIOR,
                "An unsafe consumption update is rejected without mutating state.",
                (
                    "cpp/tests/physics/test_legacy_reaction_safety.cpp::"
                    "specializedWrappersEnforceReactionAwarePositivity",
                ),
            ),
        ),
        exclusions=("variable diffusivity", "advection"),
        warnings=(
            "No measured convergence order is registered for this specialization.",
        ),
    ),
    SolverContract(
        contract_id="reaction.masked_michaelis_menten",
        title="Masked Michaelis-Menten reaction-diffusion",
        native_symbols=("MaskedMichaelisMentenReactionDiffusionSolver",),
        equation=(
            "du/dt = D laplacian(u) - Vmax*u/(Km+u) outside mask; u=pinned_value in mask"
        ),
        unknowns=("scalar field u",),
        locations=("StructuredMesh nodes with a byte-valued pinning mask",),
        input_units=_SCALAR_INPUT_UNITS
        + (
            ("vmax", "field/time"),
            ("km", "field"),
            ("pinned_value", "field"),
            ("mask", "1"),
        ),
        output_units=_SCALAR_OUTPUT_UNITS,
        supported_dimensions=("1D", "2D"),
        supported_terms=(
            "constant diffusion",
            "Michaelis-Menten consumption",
            "internally pinned nodes",
        ),
        supported_boundary_conditions=_LEGACY_SCALAR_BOUNDARIES,
        numerical_method="Forward Euler with nodal pinning reapplied after each step.",
        stability_policy=(
            "The diffusion bound is enforced and complete unpinned candidates must remain "
            "finite and nonnegative; pinning-interface accuracy is not automatically certified."
        ),
        convergence_policy="No direct numerical test or convergence study is registered.",
        evidence=(
            _API_EXPORT_EVIDENCE,
            EvidenceRecord(
                EvidenceLevel.BEHAVIOR,
                "An unsafe unpinned update is rejected without mutating the public state.",
                (
                    "cpp/tests/physics/test_legacy_reaction_safety.cpp::"
                    "specializedWrappersEnforceReactionAwarePositivity",
                ),
            ),
        ),
        exclusions=("variable diffusivity", "advection"),
        warnings=(
            "No measured convergence order is registered for this specialization.",
            "Pinned nodes represent a prescribed internal value, not a resolved vessel model.",
        ),
    ),
    SolverContract(
        contract_id="reaction.constant_source_specialized",
        title="Specialized constant-source reaction-diffusion",
        native_symbols=("ConstantSourceReactionDiffusionSolver",),
        equation="du/dt = D laplacian(u) + S",
        unknowns=("scalar field u",),
        locations=("StructuredMesh nodes",),
        input_units=_SCALAR_INPUT_UNITS + (("source_rate", "field/time"),),
        output_units=_SCALAR_OUTPUT_UNITS,
        supported_dimensions=("1D", "2D"),
        supported_terms=("constant diffusion", "uniform constant source"),
        supported_boundary_conditions=_LEGACY_SCALAR_BOUNDARIES,
        numerical_method="Centred diffusion and constant source with Forward Euler.",
        stability_policy=(
            "The diffusion bound is checked. A constant source adds no linear stability "
            "restriction, but complete concentration candidates must remain finite and "
            "nonnegative before mutation."
        ),
        convergence_policy="No direct numerical test is registered for this class.",
        evidence=(
            _API_EXPORT_EVIDENCE,
            EvidenceRecord(
                EvidenceLevel.BEHAVIOR,
                "A constant sink that would create a negative concentration is rejected "
                "without mutating state.",
                (
                    "cpp/tests/physics/test_legacy_reaction_safety.cpp::"
                    "specializedWrappersEnforceReactionAwarePositivity",
                ),
            ),
        ),
        exclusions=(
            "spatially varying source",
            "variable diffusivity",
        ),
        warnings=(
            "No measured convergence order is registered for this specialization.",
        ),
    ),
    SolverContract(
        contract_id="reaction.multispecies",
        title="Conservative multi-species reaction-diffusion",
        native_symbols=("MultiSpeciesSolver",),
        equation="dc_i/dt = D_i laplacian(c_i) + R_i(c_1,...,c_N,x,y,t)",
        unknowns=("N scalar species fields",),
        locations=("vertex-centred control volumes on StructuredMesh",),
        input_units=(
            ("concentration_i", "species-specific concentration unit"),
            ("diffusivity_i", "length^2/time"),
            ("reaction_rate_i", "concentration_i/time"),
            ("coordinate", "length"),
            ("time", "time"),
        ),
        output_units=(
            ("concentration_i", "same as species input"),
            ("total_mass_i", "concentration_i*length^dimension"),
            ("time", "time"),
        ),
        supported_dimensions=("1D", "2D"),
        supported_terms=(
            "species-specific constant diffusion",
            "local coupled reaction callback",
            "Lotka-Volterra, SIR, SEIR, Brusselator, competitive inhibition, and enzyme cascade models",
        ),
        supported_boundary_conditions=_SCALAR_BOUNDARIES,
        numerical_method=(
            "Conservative nodal finite-volume diffusion and local reactions with "
            "Forward Euler."
        ),
        stability_policy=(
            "max_stable_time_step certifies diffusion only. solve_until subdivides to "
            "that limit, while reaction stability/positivity remains model dependent; "
            "nonfinite or negative updates fail atomically."
        ),
        convergence_policy=(
            "A diffusion eigenmode limit and conservation are tested, but no general "
            "order claim is made for arbitrary coupled kinetics."
        ),
        evidence=(
            EvidenceRecord(
                EvidenceLevel.ANALYTICAL,
                "Closed diffusion conserves trapezoidal mass, an eigenmode follows its "
                "expected decay limit, SIR kinetics conserve local population, and Python "
                "callback mutations/returns are exercised.",
                (
                    "cpp/tests/physics/test_multispecies_conservative.cpp::"
                    "testDiffusionLimitAndNeumannEigenmode",
                    "cpp/tests/physics/test_multispecies_conservative.cpp::"
                    "testHomogeneousSirInvariant",
                    "python/tests/test_multi_species.py::"
                    "test_python_reaction_callback_copies_mutated_rates",
                    "python/tests/test_multi_species.py::"
                    "test_python_reaction_callback_accepts_returned_sequence",
                ),
            ),
        ),
        exclusions=(
            "cross-diffusion",
            "advection",
            "automatic stiffness handling",
            "biological calibration of built-in kinetics",
        ),
        warnings=(
            "SIR/SEIR N is a local reference in the same units as the local fields.",
            "Brusselator parameters are nondimensional unless the user supplies a scaling.",
        ),
    ),
    SolverContract(
        contract_id="reaction.gray_scott",
        title="Periodic Gray-Scott model",
        native_symbols=("GrayScottSolver",),
        equation=(
            "du/dt = Du laplacian(u)-u*v^2+f*(1-u); "
            "dv/dt = Dv laplacian(v)+u*v^2-(f+k)*v"
        ),
        unknowns=("nondimensional u", "nondimensional v"),
        locations=("periodic cell-centred nx by ny grid",),
        input_units=(
            ("u", "1"),
            ("v", "1"),
            ("Du", "grid_length^2/model_time"),
            ("Dv", "grid_length^2/model_time"),
            ("f", "1/model_time"),
            ("k", "1/model_time"),
            ("dt", "model_time"),
        ),
        output_units=(
            ("u", "1"),
            ("v", "1"),
            ("time", "model_time"),
        ),
        supported_dimensions=("2D",),
        supported_terms=("periodic diffusion", "Gray-Scott feed/kill kinetics"),
        supported_boundary_conditions=("periodic in x and y",),
        numerical_method="Periodic centred Laplacian and Forward Euler in float32.",
        stability_policy=(
            "The explicit diffusion ceiling and finite/positivity checks are enforced. "
            "Reaction accuracy may require smaller steps."
        ),
        convergence_policy=(
            "Homogeneous kinetics, grid-spacing use, periodic conservation, positivity, "
            "and deterministic execution are tested; no pattern convergence order is claimed."
        ),
        evidence=(
            EvidenceRecord(
                EvidenceLevel.INVARIANT,
                "Periodic diffusion conserves the discrete sum, homogeneous kinetics match "
                "one Forward-Euler step, and invalid/unstable states fail loudly.",
                (
                    "cpp/tests/physics/test_multispecies_gray_scott.cpp::"
                    "testPeriodicDiffusionConservesSumAndPositivity",
                    "cpp/tests/physics/test_multispecies_gray_scott.cpp::"
                    "testHomogeneousKineticsOneStep",
                    "cpp/tests/physics/test_multispecies_gray_scott.cpp::"
                    "testUnstableAndNonfiniteInputsFailLoudly",
                ),
            ),
        ),
        exclusions=(
            "nonperiodic boundaries",
            "3D patterns",
            "dimensional chemical calibration",
        ),
        warnings=("Pattern resemblance is not evidence of a biological mechanism.",),
    ),
    SolverContract(
        contract_id="flow.darcy",
        title="Steady Darcy porous flow",
        native_symbols=("DarcyFlowSolver",),
        equation="v = -kappa*grad(p); div(kappa*grad(p)) = 0",
        unknowns=("pressure p", "derived superficial velocity v"),
        locations=("StructuredMesh nodes",),
        input_units=(
            ("pressure", "Pa"),
            ("hydraulic_conductivity", "m^2/(Pa*s)"),
            ("coordinate", "m"),
            ("outward_pressure_derivative", "Pa/m"),
        ),
        output_units=(
            ("pressure", "Pa"),
            ("velocity", "m/s"),
            ("residual", "maximum pressure fixed-point defect [Pa]"),
        ),
        supported_dimensions=("2D",),
        supported_terms=(
            "scalar or nodal hydraulic mobility K/mu",
            "internally pinned pressure nodes",
        ),
        supported_boundary_conditions=(
            "Dirichlet pressure",
            "Neumann outward-normal pressure derivative",
        ),
        numerical_method="SOR pressure solve followed by a Darcy-law velocity gradient.",
        stability_policy=(
            "A pressure gauge supplied by at least one Dirichlet or internal-pressure "
            "constraint is required. Invalid or singular configurations and exhausted "
            "steady iterations throw instead of returning an unconverged field."
        ),
        convergence_policy=(
            "Uniform pressure/velocity and a face-aligned two-material flux balance are "
            "tested analytically. A node-aligned discontinuous interface has a measured "
            "approximately first-order pressure-error sequence; no broader order is implied."
        ),
        evidence=(
            EvidenceRecord(
                EvidenceLevel.ANALYTICAL,
                "Uniform linear pressure and velocity match Darcy's law, and a face-aligned "
                "two-material case preserves one normal flux and the analytical pressure drop.",
                (
                    "cpp/tests/physics/test_darcy_science.cpp::"
                    "uniformPressureDropMatchesDarcyLaw",
                    "cpp/tests/physics/test_darcy_science.cpp::"
                    "layeredMediumConservesNormalFaceFlux",
                ),
            ),
            EvidenceRecord(
                EvidenceLevel.CONVERGENCE,
                "The node-aligned discontinuous-conductivity pressure error decreases at the "
                "measured first-order rate for the registered refinement case.",
                (
                    "cpp/tests/physics/test_darcy_science.cpp::"
                    "discontinuousInterfaceRefinesAtMeasuredFirstOrder",
                ),
            ),
            EvidenceRecord(
                EvidenceLevel.BEHAVIOR,
                "Outward-gradient signs, missing-gauge rejection, and forced nonconvergence "
                "are exercised explicitly.",
                (
                    "cpp/tests/physics/test_darcy_science.cpp::"
                    "outwardGradientSignAndUnitsAreExplicit",
                    "cpp/tests/physics/test_darcy_science.cpp::"
                    "singularAndUnconvergedProblemsFailLoudly",
                ),
            ),
        ),
        exclusions=(
            "transient storage",
            "compressibility",
            "non-Darcy inertia",
            "separate viscosity input",
        ),
        warnings=(
            "kappa is hydraulic mobility K/mu [m^2/(Pa*s)], not hydraulic conductivity "
            "and not intrinsic permeability K.",
            "The legacy set_neumann flux keyword denotes dp/dn; Python callers should prefer "
            "set_outward_pressure_gradient.",
            "At least one pressure constraint is required to fix the all-Neumann nullspace.",
            "Registered discontinuous-interface refinement is first order and is not a smooth-"
            "media order claim.",
        ),
    ),
    SolverContract(
        contract_id="flow.stokes",
        title="Steady incompressible Stokes flow",
        native_symbols=("StokesSolver",),
        equation="-grad(p) + mu*laplacian(v) + f = 0; div(v) = 0",
        unknowns=("velocity v", "pressure p"),
        locations=(
            "collocated velocity components and pressure at StructuredMesh nodes",
        ),
        input_units=(
            ("viscosity", "Pa*s"),
            ("velocity", "m/s"),
            ("pressure", "Pa"),
            ("body_force", "N/m^3"),
            ("coordinate", "m"),
        ),
        output_units=(
            ("velocity", "m/s"),
            ("pressure", "Pa"),
            ("divergence", "1/s"),
            ("momentum_residual", "N/m^3"),
        ),
        supported_dimensions=("2D",),
        supported_terms=(
            "viscous momentum",
            "pressure",
            "uniform scalar or coordinate-dependent callback body force",
            "exact sealed all-no-slip hydrostatic equilibrium for the scalar-force overload",
        ),
        supported_boundary_conditions=(
            "no-slip velocity",
            "constant Dirichlet velocity",
            "constant inflow velocity",
            "zero outward-normal velocity-gradient outflow",
        ),
        numerical_method=(
            "Collocated centred-difference momentum equations with Gauss-Seidel "
            "relaxation and a SIMPLE-like pressure-correction iteration. A sealed all-no-slip "
            "domain with a body force supplied through the scalar overload uses the exact "
            "zero-velocity hydrostatic solution p=f dot x with a zero-nodal-mean gauge."
        ),
        stability_policy=(
            "No time step is present. Invalid/non-finite states and exhausted steady "
            "iterations throw; a returned result exposes momentum residual and maximum "
            "divergence and is converged."
        ),
        convergence_policy=(
            "The sealed uniform-force pressure and zero velocity are checked against the exact "
            "hydrostatic equilibrium. Plane Poiseuille velocity is compared with the analytical "
            "profile. A Python grid sequence requires each sampled error to remain below 0.5% "
            "but does not assert monotonic decrease or an observed order. No broader pressure-"
            "accuracy or spatial-order claim is registered."
        ),
        evidence=(
            EvidenceRecord(
                EvidenceLevel.ANALYTICAL,
                "A uniform scalar body force in a sealed all-no-slip domain is balanced by "
                "p=f dot x with a deterministic zero-nodal-mean gauge and exactly zero velocity.",
                (
                    "cpp/tests/physics/test_stokes.cpp::"
                    "testSealedUniformForceHydrostaticEquilibrium",
                    "python/tests/test_stokes.py::"
                    "test_uniform_body_force_is_balanced_by_pressure",
                ),
            ),
            EvidenceRecord(
                EvidenceLevel.ANALYTICAL,
                "Plane Poiseuille flow is checked for analytical velocity accuracy, "
                "momentum residual, and divergence; the Python grid sequence checks that "
                "velocity error remains below 0.5% without making an order claim.",
                (
                    "cpp/tests/physics/test_stokes.cpp::"
                    "testPlanePoiseuilleAccuracyAndConservation",
                    "python/tests/test_stokes.py::test_channel_accuracy_across_refinement",
                ),
            ),
        ),
        exclusions=("inertia", "transient flow", "non-Newtonian viscosity", "3D flow"),
        warnings=(
            "Applicability requires a creeping-flow regime; Reynolds number alone is diagnostic.",
            "The collocated pressure field has no Rhie-Chow or equivalent checkerboard "
            "stabilization and must not be interpreted as a staggered MAC discretization.",
            "Only the scalar set_body_force overload is classified as spatially uniform; "
            "callback forces always use the generic iterative path.",
            "Outflow means zero outward-normal velocity gradient, not traction-free or "
            "prescribed-pressure outflow; registered inflow/outflow evidence is limited to a "
            "flux-compatible uniform plug flow.",
        ),
    ),
    SolverContract(
        contract_id="flow.navier_stokes",
        title="Bounded incompressible Navier-Stokes flow",
        native_symbols=("NavierStokesSolver",),
        equation=(
            "dv/dt + (v dot grad)v = -(1/rho)grad(p) + nu*laplacian(v) + f/rho; div(v)=0"
        ),
        unknowns=("velocity v", "pressure p"),
        locations=("MAC face velocities in packed arrays", "cell-centred pressure"),
        input_units=(
            ("density", "kg/m^3"),
            ("viscosity", "Pa*s"),
            ("velocity", "m/s"),
            ("pressure", "Pa"),
            ("body_force", "N/m^3"),
            ("time", "s"),
        ),
        output_units=(
            ("velocity", "m/s"),
            ("pressure", "Pa"),
            ("divergence", "1/s"),
            ("time", "s"),
        ),
        supported_dimensions=("2D",),
        supported_terms=(
            "inertia",
            "pressure projection",
            "viscosity",
            "uniform or prescribed finite spatial body-force callbacks",
            "UPWIND or CENTRAL convection",
        ),
        supported_boundary_conditions=(
            "no-slip",
            "flux-compatible constant Dirichlet velocity",
        ),
        numerical_method=(
            "Explicit finite-volume MAC predictor with a compatible pressure projection "
            "and SOR pressure solve."
        ),
        stability_policy=(
            "Fixed steps above the convective/diffusive bound are rejected; adaptive mode "
            "recomputes the ceiling. Pressure non-convergence and incompatible boundary "
            "flux fail loudly."
        ),
        convergence_policy=(
            "A smooth, wall-compatible manufactured central-convection velocity field "
            "measures approximately second-order spatial convergence over three square "
            "MAC grids with first-order time error suppressed by dt=O(h^3). Projection "
            "divergence, exact-time semantics, uniform flow, and failure modes are tested "
            "separately."
        ),
        evidence=(
            EvidenceRecord(
                EvidenceLevel.CONVERGENCE,
                "An independently forced incompressible manufactured velocity field "
                "converges over three grids; exact uniform flow and compatible projection "
                "invariants exercise separate paths.",
                (
                    "cpp/tests/physics/test_navier_stokes.cpp::"
                    "manufacturedSteadyVelocityConverges",
                    "cpp/tests/physics/test_navier_stokes.cpp::"
                    "compatibleProjectionReducesDivergence",
                    "python/tests/test_navier_stokes.py::"
                    "test_uniform_velocity_is_an_exact_navier_stokes_solution",
                ),
            ),
        ),
        exclusions=(
            "3D flow",
            "turbulence closure",
            "open, traction, profile-inlet, QUICK, or HYBRID configurations",
            "non-Newtonian viscosity",
        ),
        warnings=("Intended for resolved laminar flows, not turbulence prediction.",),
    ),
    SolverContract(
        contract_id="membrane.single_layer",
        title="Steady single-layer membrane diffusion",
        native_symbols=("MembraneDiffusion1DSolver",),
        equation="d/dx(D*dC/dx)=0; J = (D*Phi*H/L)*(C_left-C_right)",
        unknowns=("intramembrane concentration C", "steady flux J"),
        locations=("uniform 1D membrane nodes",),
        input_units=(
            ("thickness", "m"),
            ("diffusivity", "m^2/s"),
            ("partition_coefficient", "1"),
            ("concentration", "amount/m^3"),
            ("solute_and_pore_radius", "same length unit"),
        ),
        output_units=(
            ("concentration", "amount/m^3"),
            ("flux", "amount/(m^2*s)"),
            ("permeability", "m/s"),
            ("effective_diffusivity", "m^2/s"),
        ),
        supported_dimensions=("steady 1D",),
        supported_terms=(
            "constant membrane diffusivity",
            "partitioning",
            "optional Renkin pore hindrance",
        ),
        supported_boundary_conditions=(
            "prescribed external concentration on both sides",
        ),
        numerical_method="Closed-form steady resistance with a sampled linear membrane profile.",
        stability_policy="No time integration or stability restriction.",
        convergence_policy=(
            "Closed-form resistance, equilibrium, reversed-gradient sign, and invalid domains "
            "are tested; no transient or grid-convergence claim applies."
        ),
        evidence=(
            EvidenceRecord(
                EvidenceLevel.ANALYTICAL,
                "Flux, permeability, and apparent P*L coefficient match the single-layer "
                "resistance relation; equilibrium and sign reversal are checked.",
                (
                    "cpp/tests/physics/test_electrochem_membrane.cpp::"
                    "testSingleLayerResistanceAndApparentCoefficient",
                    "cpp/tests/physics/test_electrochem_membrane.cpp::"
                    "testReverseGradientReversesFlux",
                ),
            ),
        ),
        exclusions=(
            "transient storage",
            "solvent drag",
            "electromigration",
            "concentration-dependent material properties",
        ),
        warnings=(
            "Renkin hindrance is a limited pore correlation, not a universal biological correction.",
            "effective_diffusivity is the external-gradient equivalent P*L.",
        ),
    ),
    SolverContract(
        contract_id="membrane.multilayer",
        title="Steady multilayer membrane diffusion",
        native_symbols=("MultiLayerMembraneSolver",),
        equation="J = (C_left-C_right)/sum_i[L_i/(D_i*K_i)]",
        unknowns=("piecewise intramembrane concentration", "common steady flux"),
        locations=("one-dimensional ordered membrane layers",),
        input_units=(
            ("layer_thickness", "m"),
            ("layer_diffusivity", "m^2/s"),
            ("partition_coefficient", "1"),
            ("concentration", "amount/m^3"),
        ),
        output_units=(
            ("concentration", "amount/m^3"),
            ("flux", "amount/(m^2*s)"),
            ("permeability", "m/s"),
            ("effective_diffusivity", "m^2/s"),
        ),
        supported_dimensions=("steady 1D",),
        supported_terms=("series layer resistance", "layer partitioning"),
        supported_boundary_conditions=(
            "prescribed external concentration on both sides",
        ),
        numerical_method="Closed-form series resistance with piecewise linear profiles.",
        stability_policy="No time integration or stability restriction.",
        convergence_policy=(
            "Layer resistance, interfacial reference activity, equilibrium, and flux sign "
            "are tested against closed-form balances."
        ),
        evidence=(
            EvidenceRecord(
                EvidenceLevel.ANALYTICAL,
                "Layer interfaces preserve reference activity and the common flux matches "
                "the summed resistance relation.",
                (
                    "cpp/tests/physics/test_electrochem_membrane.cpp::"
                    "testLayerInterfacesPreserveReferenceActivity",
                    "python/tests/test_membrane_diffusion.py::"
                    "test_multilayer_resistance_series",
                ),
            ),
        ),
        exclusions=("transient storage", "interfacial kinetics", "solvent drag"),
        warnings=(
            "Layer order and partition conventions must match the physical model.",
        ),
    ),
    SolverContract(
        contract_id="applications.tumor_drug_delivery",
        title="Reduced tumor drug-delivery model",
        native_symbols=("TumorDrugDeliverySolver",),
        equation=(
            "Darcy pressure/velocity plus explicit free-drug diffusion-advection, "
            "vascular source, irreversible binding, and cellular uptake compartments"
        ),
        unknowns=(
            "interstitial pressure",
            "free drug concentration",
            "bound drug concentration",
            "cellular drug concentration",
        ),
        locations=("StructuredMesh nodes and supplied tumor mask",),
        input_units=(
            ("pressure", "Pa"),
            ("hydraulic_conductivity", "m^2/(Pa*s)"),
            ("diffusivity", "m^2/s"),
            ("vessel_wall_solute_permeability", "m/s"),
            ("vascular_surface_area_density", "1/m"),
            ("binding_and_uptake_rate", "1/s"),
            ("concentration", "amount/m^3"),
            ("time", "s"),
        ),
        output_units=(
            ("pressure", "Pa"),
            ("concentration", "amount/m^3"),
            ("time", "s"),
        ),
        supported_dimensions=("2D",),
        supported_terms=(
            "SOR pressure field",
            "pressure-driven convection",
            "nodal diffusion",
            "vascular solute exchange",
            "irreversible first-order binding and uptake",
        ),
        supported_boundary_conditions=(
            "fixed outer pressure",
            "clamped tumor-mask pressure",
            "model-defined drug boundaries",
        ),
        numerical_method=(
            "SOR pressure solve followed by explicit conservative compartment transport."
        ),
        stability_policy=(
            "Transport rejects unstable or dimensionally invalid steps and lands on requested "
            "save times; pressure convergence is reported through the solve tolerance."
        ),
        convergence_policy=(
            "Pressure symmetry/bounds and compartment/source/outflow balances are tested. "
            "No patient-level, experimental, or grid-order validation is registered."
        ),
        evidence=(
            EvidenceRecord(
                EvidenceLevel.INVARIANT,
                "Pressure bounds/symmetry and conservative compartment, vascular-source, "
                "and pressure-driven outflow balances are checked.",
                (
                    "cpp/tests/physics/test_tumor_drug_delivery_science.cpp::"
                    "pressureSolveRespectsBoundsSymmetryAndConvergence",
                    "cpp/tests/physics/test_tumor_drug_delivery_science.cpp::"
                    "compartmentTransfersAndVascularSourceCloseTheMassBalance",
                    "cpp/tests/physics/test_tumor_drug_delivery_science.cpp::"
                    "pressureDrivenOutflowIsAccountedConservatively",
                ),
            ),
        ),
        exclusions=(
            "Starling fluid filtration",
            "solvent drag",
            "reversible or saturable binding",
            "systemic pharmacokinetics",
            "patient-specific calibration",
        ),
        warnings=(
            "Clamped tumor pressure represents an unresolved solute-free fluid source.",
            "This reduced model is not a clinical predictor.",
        ),
    ),
    SolverContract(
        contract_id="applications.bioheat_cryotherapy",
        title="Pennes bioheat with apparent phase change",
        native_symbols=("BioheatCryotherapySolver",),
        equation=(
            "rho*c_eff(T)*dT/dt = div(k(T)grad(T)) + "
            "omega*rho_b*c_b*(T_arterial-T) + Q_met"
        ),
        unknowns=(
            "absolute temperature T",
            "Arrhenius heat-injury integral",
            "frozen fraction",
        ),
        locations=("StructuredMesh nodes with an embedded probe mask",),
        input_units=(
            ("temperature", "K"),
            ("density", "kg/m^3"),
            ("specific_heat", "J/(kg*K)"),
            ("thermal_conductivity", "W/(m*K)"),
            ("perfusion", "1/s"),
            ("metabolic_source", "W/m^3"),
            ("latent_heat", "J/kg"),
            ("arrhenius_A", "1/s"),
            ("activation_energy", "J/mol"),
            ("time", "s"),
        ),
        output_units=(
            ("temperature", "K"),
            ("frozen_fraction", "1"),
            ("arrhenius_integral", "1"),
            ("time", "s"),
        ),
        supported_dimensions=("2D",),
        supported_terms=(
            "temperature-dependent conduction",
            "apparent heat capacity",
            "Pennes perfusion",
            "metabolic heating",
            "fixed-temperature probe nodes",
            "Arrhenius heat-injury diagnostic",
        ),
        supported_boundary_conditions=(
            "fixed outer temperature",
            "fixed probe-mask temperature",
        ),
        numerical_method="Explicit nodal heat balance with apparent heat capacity.",
        stability_policy=(
            "maximum_stable_time_step_s reports the explicit thermal ceiling; larger, "
            "nonfinite, or nonphysical configurations fail before mutation."
        ),
        convergence_policy=(
            "A linear zero-perfusion heat eigenmode measures second-order spatial "
            "convergence with Forward-Euler error suppressed by dt=O(h^3), and measures "
            "first-order Forward-Euler time convergence against its exact semi-discrete "
            "mode. Latent heat, Pennes equilibrium, save times, and stability rejection "
            "are tested separately."
        ),
        evidence=(
            EvidenceRecord(
                EvidenceLevel.CONVERGENCE,
                "Independent heat eigenmodes measure scoped spatial and temporal order; "
                "uniform Pennes equilibrium, latent heat, and source signs remain invariant "
                "checks.",
                (
                    "cpp/tests/physics/test_bioheat_science.cpp::"
                    "smoothHeatEigenmodeHasSecondOrderSpatialConvergence",
                    "cpp/tests/physics/test_bioheat_science.cpp::"
                    "explicitEulerTimeIntegrationConvergesAtFirstOrder",
                    "cpp/tests/physics/test_bioheat_science.cpp::"
                    "apparentCapacityHasCorrectUnitsAndLatentIntegral",
                    "cpp/tests/physics/test_bioheat_science.cpp::"
                    "uniformPennesEquilibriumIsPreserved",
                    "cpp/tests/physics/test_bioheat_science.cpp::"
                    "perfusionSignAndUnitsRestoreTowardArterialTemperature",
                ),
            ),
        ),
        exclusions=(
            "probe contact resistance or coolant flow",
            "conjugate probe physics",
            "cryogenic cell-death prediction",
            "patient-specific calibration",
        ),
        warnings=(
            "All API temperatures are absolute kelvin.",
            "The Arrhenius integral is a heat-injury diagnostic, not a cryoinjury model.",
        ),
    ),
    SolverContract(
        contract_id="electrochem.nernst_planck",
        title="Single-ion prescribed-potential Nernst-Planck transport",
        native_symbols=("NernstPlanckSolver",),
        equation=("J = -D[grad(c) + z*F*c*grad(phi)/(R*T)]; dc/dt = -div(J)"),
        unknowns=("ion concentration c",),
        locations=("vertex-centred control volumes on StructuredMesh",),
        input_units=(
            ("concentration", "mol/m^3"),
            ("diffusivity", "m^2/s"),
            ("potential", "V"),
            ("electric_field", "V/m"),
            ("temperature", "K"),
            ("time", "s"),
        ),
        output_units=(
            ("concentration", "mol/m^3"),
            ("current_density", "A/m^2"),
            ("potential", "V"),
            ("time", "s"),
        ),
        supported_dimensions=("1D", "2D"),
        supported_terms=(
            "ideal dilute diffusion",
            "electromigration in prescribed potential",
            "temperature-aware Einstein mobility",
        ),
        supported_boundary_conditions=(
            "Dirichlet concentration",
            "prescribed outward total molar flux",
        ),
        numerical_method=(
            "Conservative fitted diffusion-drift face flux with explicit time stepping."
        ),
        stability_policy=(
            "A positivity bound for the fitted homogeneous operator is exposed and checked; "
            "recommended_time_step applies a user-selected safety factor."
        ),
        convergence_policy=(
            "The zero-potential diffusion limit measures second-order spatial convergence "
            "against an independent Neumann cosine mode with first-order time error "
            "suppressed by dt=O(h^3). Discrete Boltzmann equilibrium, conservation, flux "
            "balance, migration direction, and limiting potentials are tested separately; "
            "no temporal-order claim is registered."
        ),
        evidence=(
            EvidenceRecord(
                EvidenceLevel.CONVERGENCE,
                "A Neumann diffusion eigenmode measures scoped second-order spatial "
                "convergence with time error suppressed; Boltzmann equilibrium, prescribed "
                "flux balance, and migration direction supply independent "
                "analytical/invariant evidence.",
                (
                    "cpp/tests/physics/test_electrochem_nernst_planck.cpp::"
                    "testDiffusionLimitConvergesAgainstNeumannEigenmode",
                    "cpp/tests/physics/test_electrochem_nernst_planck.cpp::"
                    "testBoltzmannEquilibriumHasZeroFlux",
                    "cpp/tests/physics/test_electrochem_nernst_planck.cpp::"
                    "testPrescribedOutwardFluxClosesMassBalance",
                    "cpp/tests/physics/test_electrochem_nernst_planck.cpp::"
                    "testValenceControlsMigrationDirection",
                ),
            ),
        ),
        exclusions=(
            "Poisson electrostatics",
            "activity or finite-size corrections",
            "fluid coupling",
            "membrane gating or capacitance",
        ),
        warnings=(
            "Potential is prescribed; this is not Poisson-Nernst-Planck.",
            "The default wall condition is zero total molar flux, not zero concentration gradient.",
        ),
    ),
    SolverContract(
        contract_id="electrochem.multi_ion",
        title="Independent multi-ion prescribed-potential transport",
        native_symbols=("MultiIonSolver",),
        equation=(
            "For each i: J_i=-D_i[grad(c_i)+z_i*F*c_i*grad(phi)/(R*T)]; "
            "dc_i/dt=-div(J_i)"
        ),
        unknowns=("one concentration field per ion",),
        locations=("vertex-centred control volumes on StructuredMesh",),
        input_units=(
            ("concentration_i", "mol/m^3"),
            ("diffusivity_i", "m^2/s"),
            ("potential", "V"),
            ("electric_field", "V/m"),
            ("temperature", "K"),
            ("time", "s"),
        ),
        output_units=(
            ("concentration_i", "mol/m^3"),
            ("charge_density", "C/m^3"),
            ("potential", "V"),
            ("time", "s"),
        ),
        supported_dimensions=("1D", "2D"),
        supported_terms=(
            "independent ideal dilute diffusion",
            "independent electromigration in one prescribed potential",
        ),
        supported_boundary_conditions=(
            "species Dirichlet concentration",
            "species prescribed outward total molar flux",
        ),
        numerical_method=(
            "One conservative fitted explicit diffusion-drift operator per species."
        ),
        stability_policy=(
            "The exposed bound is the minimum fitted-operator positivity ceiling over species."
        ),
        convergence_policy=(
            "Independent species conservation and invalid/unsupported coupling rejection are "
            "tested; no coupled electrostatic or convergence-order claim is registered."
        ),
        evidence=(
            EvidenceRecord(
                EvidenceLevel.INVARIANT,
                "Multiple sealed species conserve independently in a prescribed potential, "
                "and enabling unsupported electroneutral coupling fails loudly.",
                (
                    "cpp/tests/physics/test_electrochem_nernst_planck.cpp::"
                    "testMultiIonSpeciesConserveIndependently",
                    "cpp/tests/physics/test_electrochem_nernst_planck.cpp::"
                    "testUnsupportedCouplingAndInvalidInputsFailLoudly",
                ),
            ),
        ),
        exclusions=(
            "Poisson electrostatics",
            "electroneutral coupling",
            "cross-diffusion or chemical reactions",
            "activity corrections",
        ),
        warnings=(
            "Species share a prescribed potential but are otherwise uncoupled.",
            "Default species walls are sealed with zero total molar flux.",
        ),
    ),
)


_PYTHON_NUMERICAL_CONTRACTS: Final[tuple[PythonNumericalContract, ...]] = (
    PythonNumericalContract(
        contract_id="python.canonical.adapters",
        title="Friendly canonical transport adapters and checkpoint orchestration",
        module="biotransport.run",
        public_symbols=("solve", "run", "run_checkpoints", "CheckpointResult"),
        category="native solver adapter",
        backend=PythonBackend.NATIVE_ADAPTER,
        mathematical_scope=(
            "solve/run configure the native canonical conservative scalar transport "
            "solver; run_checkpoints repeats uniform-diffusion native solves over "
            "explicitly partitioned time intervals."
        ),
        numerical_method=(
            "Argument normalization followed by solve_transport. Checkpoints carry "
            "the final field into the next native segment and retain per-segment "
            "diagnostics."
        ),
        failure_policy=(
            "Ambiguous aliases, non-real/non-finite controls, invalid counts, malformed "
            "checkpoint fields, and cumulative step exhaustion raise before misleading "
            "results are returned."
        ),
        evidence=(
            EvidenceRecord(
                EvidenceLevel.INVARIANT,
                "The adapters preserve exact final-time/native diagnostics, and "
                "checkpoint partitioning matches a one-shot run when the requested "
                "step divides every segment.",
                (
                    "python/tests/test_transport_api.py::"
                    "test_requested_step_is_honored_and_last_step_lands_exactly",
                    "python/tests/test_run_api.py::"
                    "test_run_checkpoints_matches_one_shot_when_dt_divides_every_segment",
                    "python/tests/test_run_api.py::"
                    "test_solve_rejects_ambiguous_or_nonfinite_options",
                ),
            ),
        ),
        disposition=(
            "Retain. The canonical solve is a thin native adapter and now forwards "
            "save_times to the native solve-at-times schedule; run_checkpoints is "
            "deprecated in favour of solve(save_times=...)."
        ),
        exclusions=(
            "run_checkpoints with reactions, advection, or variable diffusivity",
            "bitwise equivalence between segmented and one-shot automatic stepping",
        ),
        warnings=(
            "Checkpoint boundaries can change shortened/automatic step partitioning.",
            "Per-segment diagnostic times are durations; mapping keys are absolute times.",
        ),
    ),
    PythonNumericalContract(
        contract_id="python.native_adapter.stepping",
        title="Shared solve_until lifecycle for native stepping solvers",
        module="biotransport.stepping",
        public_symbols=(
            "solve_until",
            "StepDiagnostics",
            "registered_stepping_classes",
        ),
        category="native solver adapter",
        backend=PythonBackend.NATIVE_ADAPTER,
        mathematical_scope=(
            "Time orchestration only: partitions [solver.time(), end_time] at the "
            "requested save times and asks the registered native solver to advance "
            "each segment with equal substeps no larger than time_step. The "
            "discretization, stability certificate, and boundary semantics remain "
            "those of the native class and its own contract."
        ),
        numerical_method=(
            "Per segment, n = ceil(duration / time_step) equal native steps (or the "
            "class's own native solve_until where it exists); the native clock must "
            "agree with the target to within 64 ulp plus 4 ulp per step taken. Fluent "
            "dirichlet/neumann/robin/boundary/"
            "outward_flux verbs forward to the class's setters without reinterpretation."
        ),
        failure_policy=(
            "Unregistered classes, end times before the current clock, non-finite or "
            "non-positive steps, unordered save times, a drifted native clock, and "
            "boundary kinds a class does not implement raise; an automatic time_step "
            "is offered only when the native class exposes its own explicit stability "
            "certificate, otherwise time_step is required."
        ),
        evidence=(
            EvidenceRecord(
                EvidenceLevel.INVARIANT,
                "Every registered class lands exactly on end_time, matches manual "
                "solve(dt, n) stepping, records snapshots at the requested times, and "
                "refuses unsupported boundary kinds; classes without a certificate "
                "require an explicit time_step.",
                (
                    "python/tests/test_stepping.py::test_solve_until_lands_exactly_and_matches_manual_stepping",
                    "python/tests/test_stepping.py::test_save_times_record_snapshots_at_requested_clocks",
                    "python/tests/test_stepping.py::test_classes_without_a_certificate_require_time_step",
                    "python/tests/test_stepping.py::test_boundary_verbs_forward_or_refuse",
                ),
            ),
        ),
        disposition=(
            "Retain as the Python face of every native stepping solver. Native "
            "solve_until/report structs may replace the Python orchestration later "
            "without changing the vocabulary."
        ),
        exclusions=(
            "any stability, accuracy, or conservation claim beyond the native class's own contract",
            "automatic time_step for classes without an explicit stability certificate",
            "snapshots of secondary fields (only the primary field is recorded)",
        ),
        warnings=(
            "Equal substeps per segment can differ from a one-shot native run when the "
            "requested step does not divide a segment.",
            "Result.diagnostics is a StepDiagnostics record, not the canonical SolveDiagnostics.",
        ),
    ),
    PythonNumericalContract(
        contract_id="python.legacy.adaptive_diffusion",
        title="Legacy adaptive 1D diffusion controller",
        module="biotransport.adaptive",
        public_symbols=(
            "AdaptiveTimeStepper",
            "AdaptiveTimeStepperConfig",
            "AdaptiveResult",
            "solve_adaptive",
        ),
        category="legacy time integrator",
        backend=PythonBackend.MIXED,
        mathematical_scope=(
            "du/dt = D*d2u/dx2 on a uniform 1D mesh with constant non-negative "
            "diffusivity and fixed Dirichlet endpoint values."
        ),
        numerical_method=(
            "Python step-doubling controller around the native DiffusionSolver "
            "Forward-Euler update; accepted state uses two half steps."
        ),
        failure_policy=(
            "Unsupported physics, invalid controller values, non-finite estimates, "
            "infeasible minimum steps, rejection exhaustion, and maximum-step exhaustion "
            "fail explicitly; callback state is isolated so attempted mutation cannot alter "
            "the accepted solution."
        ),
        evidence=(
            EvidenceRecord(
                EvidenceLevel.ANALYTICAL,
                "A smooth diffusion mode decays against its closed-form reference; "
                "unsupported physics and controller failure paths are adversarially tested.",
                (
                    "python/tests/test_adaptive.py::test_diffusion_decay",
                    "python/tests/test_adaptive.py::"
                    "test_callback_cannot_mutate_the_accepted_state",
                    "python/tests/test_adaptive.py::test_rejects_unrepresented_physics",
                    "python/tests/test_adaptive.py::"
                    "test_unsatisfied_tolerance_fails_after_rejecting_dt_min",
                ),
            ),
        ),
        disposition=(
            "Legacy compatibility surface. Port adaptive control into the C++ canonical "
            "solver before promoting it; otherwise deprecate after a native replacement."
        ),
        exclusions=(
            "2D or 3D meshes",
            "Neumann or Robin boundaries",
            "advection, reactions, sources, and variable diffusivity",
        ),
        warnings=(
            "This is Python orchestration and is not the default canonical solve path.",
            "The local step-doubling tolerance is not a global-error guarantee.",
        ),
    ),
    PythonNumericalContract(
        contract_id="python.legacy.time_integrators",
        title="Generic ODE stages and legacy higher-order diffusion wrappers",
        module="biotransport.time_integrators",
        public_symbols=(
            "RK4Integrator",
            "HeunIntegrator",
            "IntegrationResult",
            "integrate",
            "rk4_step",
            "heun_step",
            "euler_step",
        ),
        category="ODE reference methods and legacy diffusion adapters",
        backend=PythonBackend.MIXED,
        mathematical_scope=(
            "Standalone stages advance du/dt=f(u,t). HeunIntegrator and RK4Integrator "
            "advance only constant-diffusivity 1D diffusion with Dirichlet endpoints; "
            "integrate(method='euler') delegates to canonical native transport."
        ),
        numerical_method=(
            "Forward Euler, explicit trapezoidal Runge-Kutta (Heun), and classical RK4. "
            "Legacy diffusion right-hand sides are assembled in Python."
        ),
        failure_policy=(
            "State and stage shapes/finiteness are checked, callbacks receive isolated "
            "state, unstable requested diffusion steps are rejected rather than clipped, "
            "and Python loop length is bounded."
        ),
        evidence=(
            EvidenceRecord(
                EvidenceLevel.CONVERGENCE,
                "Generic ODE cases measure first-, second-, and fourth-order behavior, "
                "while legacy diffusion wrappers reject omitted physics and unstable steps.",
                (
                    "python/tests/test_time_integrators.py::test_first_order_convergence",
                    "python/tests/test_time_integrators.py::test_second_order_convergence",
                    "python/tests/test_time_integrators.py::test_fourth_order_convergence",
                    "python/tests/test_time_integrators.py::"
                    "test_rejects_requested_step_above_stability_limit",
                    "python/tests/test_time_integrators.py::"
                    "test_rejects_unrepresented_physics",
                ),
            ),
        ),
        disposition=(
            "Keep the generic stages as transparent reference utilities. Explicit "
            "method='euler' uses the native path; method is required so the algorithm "
            "is always an explicit choice. Port or eventually deprecate the "
            "legacy Python diffusion wrappers."
        ),
        exclusions=(
            "automatic stability certification for arbitrary ODE callbacks",
            "legacy Heun/RK4 support for non-Dirichlet or multidimensional transport",
        ),
        warnings=(
            "Formal RK order does not provide stability for an arbitrary caller ODE.",
            "Legacy diffusion wrappers are Python-loop orchestration, not performance APIs.",
        ),
    ),
    PythonNumericalContract(
        contract_id="python.native_backed.high_order",
        title="High-order finite differences and explicit Runge-Kutta references",
        module="biotransport.high_order",
        public_symbols=(
            "laplacian_2nd_order",
            "laplacian_4th_order",
            "laplacian_6th_order",
            "gradient_4th_order",
            "d2dx2",
            "ddx",
            "HighOrderDiffusionSolver",
            "HighOrderResult",
            "RungeKuttaResult",
            "integrate_explicit_runge_kutta",
            "verify_order_of_accuracy",
        ),
        category="native-backed spatial operators and callback time integration",
        backend=PythonBackend.MIXED,
        mathematical_scope=(
            "Uniform Cartesian derivative stencils, explicit high-order diffusion, and "
            "generic explicit Heun/RK4 integration over finite NumPy states."
        ),
        numerical_method=(
            "Compiled centred second/fourth/sixth-order derivative kernels with a "
            "native diffusion update. Generic Heun/RK4 stage orchestration runs in C++ "
            "while caller-defined right-hand sides cross into Python."
        ),
        failure_policy=(
            "Mesh/spacing/shape/domain checks, exact-time progress, spectral stability "
            "ceilings, isolated callbacks, and finite-stage validation are enforced."
        ),
        evidence=(
            EvidenceRecord(
                EvidenceLevel.CONVERGENCE,
                "Independent derivatives and generic ODE solutions measure their scoped "
                "spatial/temporal orders; unsafe diffusion and callback cases fail.",
                (
                    "python/tests/test_high_order.py::"
                    "test_laplacian_observed_order_against_known_derivative",
                    "python/tests/test_high_order.py::"
                    "test_runge_kutta_observed_order_for_exponential_growth",
                    "python/tests/test_high_order.py::"
                    "test_diffusion_rejects_unsafe_or_ill_defined_inputs",
                    "python/tests/test_high_order.py::"
                    "test_diffusion_callback_receives_isolated_shaped_copy",
                    "python/tests/test_high_order.py::"
                    "test_runge_kutta_callback_cannot_alias_accepted_state",
                    "python/tests/test_high_order.py::"
                    "test_runge_kutta_rejects_wrong_dimension_and_nonfinite_derivatives",
                ),
            ),
        ),
        disposition=(
            "Retain the compiled stencil, diffusion, and Runge-Kutta orchestration. Keep "
            "the generic callback RK surface as a reference API; fully native right-hand "
            "sides remain the performance path."
        ),
        exclusions=(
            "nonuniform or unstructured grids",
            "automatic arbitrary-ODE stability certification",
            "boundary closures with the same formal interior order",
            "extreme anisotropy or field/spacing dynamic range whose directional "
            "weights cannot be represented in binary64 (rejected explicitly)",
        ),
        warnings=(
            "Interior stencil order is not a blanket full-domain PDE order claim.",
            "Python callbacks can dominate runtime.",
            "Finite inputs can still exceed binary64's representable intermediate "
            "numeric domain; the native kernels fail loudly rather than omit a "
            "directional contribution.",
        ),
    ),
    PythonNumericalContract(
        contract_id="python.reference.newton",
        title="Reference Newton systems and steady nonlinear diffusion",
        module="biotransport.newton_raphson",
        public_symbols=(
            "NewtonSolverError",
            "NewtonEvaluationError",
            "NewtonLinearSolveError",
            "NewtonLineSearchError",
            "NewtonRaphsonSolver",
            "NonlinearDiffusionSolver",
            "NewtonResult",
            "ConvergenceCriterion",
            "michaelis_menten",
            "hill_kinetics",
            "bistable",
            "exponential_decay",
        ),
        category="nonlinear reference solver",
        backend=PythonBackend.PYTHON_REFERENCE,
        mathematical_scope=(
            "Generic finite-dimensional F(u)=0 and steady "
            "-div(D grad(u))+R(u)=S on supported uniform 1D/2D meshes."
        ),
        numerical_method=(
            "Dense/sparse Newton updates with analytic or scaled finite-difference "
            "Jacobians and an Armijo line search; conservative harmonic faces for "
            "variable 1D diffusivity."
        ),
        failure_policy=(
            "Evaluation, linear-solve, line-search, singularity, domain, and unsupported "
            "geometry failures are typed. Iteration exhaustion returns converged=False "
            "with diagnostics and must be checked by callers."
        ),
        evidence=(
            EvidenceRecord(
                EvidenceLevel.CONVERGENCE,
                "Manufactured Poisson and Neumann cases measure second-order convergence; "
                "singular systems and false convergence paths are tested independently.",
                (
                    "python/tests/test_newton_raphson.py::"
                    "test_poisson_solution_is_second_order",
                    "python/tests/test_newton_raphson.py::"
                    "test_neumann_boundary_is_second_order_for_a_manufactured_cubic",
                    "python/tests/test_newton_raphson.py::"
                    "test_singular_jacobian_fails_by_default",
                    "python/tests/test_newton_raphson.py::"
                    "test_max_iterations_returns_a_nonconverged_result",
                    "python/tests/test_newton_raphson.py::"
                    "test_tiny_damping_cannot_manufacture_update_convergence",
                    "python/tests/test_newton_raphson.py::"
                    "test_both_criterion_cannot_converge_on_residual_scaling_alone",
                ),
            ),
        ),
        disposition=(
            "Retain as a clearly labeled Python reference solver. Port repeated residual/"
            "Jacobian kernels to C++ before making performance claims."
        ),
        exclusions=(
            "variable diffusivity in 2D",
            "2D Neumann boundaries",
            "automatic positivity or uniqueness guarantees",
        ),
        warnings=(
            "A converged nonlinear residual does not establish uniqueness or model validity.",
            "Iteration exhaustion is diagnostic, not an exception; inspect converged.",
        ),
    ),
    PythonNumericalContract(
        contract_id="python.reference.pulsatile_diffusion",
        title="Legacy pulsatile-boundary diffusion reference",
        module="biotransport.pulsatile",
        public_symbols=(
            "PulsatileBC",
            "ConstantBC",
            "SinusoidalBC",
            "RampBC",
            "StepBC",
            "SquareWaveBC",
            "CustomBC",
            "ArterialPressureBC",
            "VenousPressureBC",
            "CardiacOutputBC",
            "RespiratoryBC",
            "DrugInfusionBC",
            "CompositeBC",
            "PulsatileResult",
            "solve_pulsatile",
            "heart_rate_to_period",
            "period_to_heart_rate",
            "sample_waveform",
        ),
        category="legacy dynamic-boundary reference solver",
        backend=PythonBackend.PYTHON_REFERENCE,
        mathematical_scope=(
            "du/dt=D*d2u/dx2 on a uniform 1D mesh with strong time-dependent "
            "Dirichlet endpoint values or static outward-derivative Neumann data."
        ),
        numerical_method=(
            "NumPy centred second differences and Forward Euler with explicit waveform "
            "evaluation at the new time level."
        ),
        failure_policy=(
            "Unsupported physics/boundaries/dimensions, unstable or non-finite steps, "
            "invalid waveforms, and excessive Python step counts fail; callback arrays "
            "are isolated from accepted solver state."
        ),
        evidence=(
            EvidenceRecord(
                EvidenceLevel.ANALYTICAL,
                "Closed-domain mass, outward Neumann signs, dynamic boundary timing, and "
                "the explicit diffusion ceiling are independently exercised.",
                (
                    "python/tests/test_pulsatile_science.py::"
                    "test_reference_solver_preserves_mass_with_zero_neumann_boundaries",
                    "python/tests/test_pulsatile_science.py::"
                    "test_static_neumann_values_are_outward_derivatives_with_correct_signs",
                    "python/tests/test_pulsatile_science.py::"
                    "test_reference_solver_applies_dynamic_dirichlet_at_the_new_time_level",
                    "python/tests/test_pulsatile_science.py::"
                    "test_reference_solver_rejects_an_unstable_step",
                ),
            ),
        ),
        disposition=(
            "Retain only as a warning-emitting compatibility/reference path until native "
            "time-dependent boundary support exists, then deprecate it."
        ),
        exclusions=(
            "advection, reactions, sources, and variable diffusivity",
            "2D/3D meshes and Robin boundaries",
            "physiological validation of bundled waveform templates",
        ),
        warnings=(
            "Every solve emits RuntimeWarning because the stepping loop is Python.",
            "Template waveforms are illustrative, not patient-specific evidence.",
        ),
    ),
    PythonNumericalContract(
        contract_id="python.workflow.convergence",
        title="Grid/time refinement and GCI-style workflow",
        module="biotransport.convergence",
        public_symbols=(
            "GridConvergenceStudy",
            "ConvergenceResult",
            "ConvergenceSolveResult",
            "compute_order_of_accuracy",
            "run_convergence_study",
            "temporal_convergence_study",
            "plot_convergence",
        ),
        category="verification workflow",
        backend=PythonBackend.WORKFLOW,
        mathematical_scope=(
            "Observed-order, Richardson extrapolation, and GCI-style summaries for "
            "caller-supplied scalar quantities and optional independent errors."
        ),
        numerical_method=(
            "Generalized unequal-refinement order solve plus log-regression utilities; "
            "no PDE is advanced by this module."
        ),
        failure_policy=(
            "Insufficient, duplicate, non-finite, oscillatory, partially reported, "
            "ill-conditioned, or non-log-plottable evidence raises instead of receiving "
            "a theoretical substitute."
        ),
        evidence=(
            EvidenceRecord(
                EvidenceLevel.ANALYTICAL,
                "Synthetic unequal-ratio sequences recover known order and degenerate or "
                "oscillatory sequences fail explicitly.",
                (
                    "python/tests/test_convergence.py::"
                    "test_analyze_solves_unequal_refinement_ratios",
                    "python/tests/test_convergence.py::"
                    "test_identical_solutions_are_indeterminate_not_theoretical",
                    "python/tests/test_convergence.py::"
                    "test_analyze_rejects_oscillatory_sequence",
                ),
            ),
        ),
        disposition=(
            "Retain in Python as scientific orchestration; solver kernels remain native."
        ),
        exclusions=(
            "automatic proof that a refinement sequence is asymptotic",
            "experimental validation or ASME V&V 20 compliance",
        ),
        warnings=(
            "Observed order is meaningful only for the supplied quantity and sequence.",
        ),
    ),
    PythonNumericalContract(
        contract_id="python.workflow.analysis",
        title="Sensitivity and uncertainty screening workflow",
        module="biotransport.analysis",
        public_symbols=(
            "EvaluationFailure",
            "LocalSensitivityResult",
            "Model",
            "ModelEvaluationError",
            "ParameterRange",
            "ParameterSweepResult",
            "RegressionScreeningResult",
            "SampleDesign",
            "UncertaintyResult",
            "parameter_sweep",
            "local_sensitivity",
            "latin_hypercube",
            "propagate_uncertainty",
            "standardized_regression_coefficients",
        ),
        category="scientific analysis workflow",
        backend=PythonBackend.WORKFLOW,
        mathematical_scope=(
            "Deterministic sweeps, central local sensitivity, independent-marginal Latin "
            "hypercubes, scalar uncertainty summaries, and standardized regression "
            "screening around a caller-supplied model."
        ),
        numerical_method=(
            "Python/NumPy orchestration and linear algebra over scalar model evaluations; "
            "the model callback may invoke native solvers."
        ),
        failure_policy=(
            "Invalid parameter domains, complex/non-finite or non-scalar callbacks, sample "
            "loss, rank deficiency, ill-conditioning, and invalid uncertainty summaries "
            "raise with sample context."
        ),
        evidence=(
            EvidenceRecord(
                EvidenceLevel.ANALYTICAL,
                "Known power-law sensitivities, stratification, analytic uncertainty "
                "moments, and exact linear screening coefficients are recovered.",
                (
                    "python/tests/test_analysis.py::"
                    "test_central_local_elasticities_match_power_law_exponents",
                    "python/tests/test_analysis.py::"
                    "test_latin_hypercube_is_reproducible_and_marginally_stratified",
                    "python/tests/test_analysis.py::"
                    "test_uncertainty_summary_is_reproducible_and_close_to_analytic_mean",
                    "python/tests/test_analysis.py::"
                    "test_standardized_regression_recovers_exact_linear_screening_model",
                ),
            ),
        ),
        disposition=(
            "Retain in Python as high-level orchestration; do not port callback scheduling "
            "without profiling a concrete workload."
        ),
        exclusions=(
            "correlated input distributions",
            "global Sobol/Morris indices",
            "calibration, causal inference, and model discrepancy",
        ),
        warnings=(
            "Screening results inherit every assumption and failure of the caller model.",
            "Independent-marginal sampling is not a joint biological prior.",
        ),
    ),
)


def _build_registry() -> Mapping[str, SolverContract]:
    by_id: dict[str, SolverContract] = {}
    symbol_owner: dict[str, str] = {}
    for contract in _CONTRACTS:
        if contract.contract_id in by_id:
            raise RuntimeError(f"duplicate solver contract id: {contract.contract_id}")
        by_id[contract.contract_id] = contract
        for symbol in contract.native_symbols:
            if symbol in symbol_owner:
                raise RuntimeError(
                    f"native symbol {symbol!r} belongs to both "
                    f"{symbol_owner[symbol]!r} and {contract.contract_id!r}"
                )
            symbol_owner[symbol] = contract.contract_id
    return MappingProxyType(by_id)


SOLVER_CONTRACTS: Final[Mapping[str, SolverContract]] = _build_registry()

_BY_SYMBOL: Final[Mapping[str, SolverContract]] = MappingProxyType(
    {symbol: contract for contract in _CONTRACTS for symbol in contract.native_symbols}
)


def _build_python_numerical_registry() -> Mapping[str, PythonNumericalContract]:
    by_id: dict[str, PythonNumericalContract] = {}
    symbol_owner: dict[str, str] = {}
    module_owner: dict[str, str] = {}
    for contract in _PYTHON_NUMERICAL_CONTRACTS:
        if contract.contract_id in by_id:
            raise RuntimeError(
                f"duplicate Python numerical contract id: {contract.contract_id}"
            )
        if contract.module in module_owner:
            raise RuntimeError(
                f"Python module {contract.module!r} belongs to both "
                f"{module_owner[contract.module]!r} and {contract.contract_id!r}"
            )
        by_id[contract.contract_id] = contract
        module_owner[contract.module] = contract.contract_id
        for symbol in contract.public_symbols:
            if symbol in symbol_owner:
                raise RuntimeError(
                    f"Python numerical symbol {symbol!r} belongs to both "
                    f"{symbol_owner[symbol]!r} and {contract.contract_id!r}"
                )
            symbol_owner[symbol] = contract.contract_id
    return MappingProxyType(by_id)


PYTHON_NUMERICAL_CONTRACTS: Final[Mapping[str, PythonNumericalContract]] = (
    _build_python_numerical_registry()
)

_PYTHON_BY_SYMBOL: Final[Mapping[str, PythonNumericalContract]] = MappingProxyType(
    {
        symbol: contract
        for contract in _PYTHON_NUMERICAL_CONTRACTS
        for symbol in contract.public_symbols
    }
)


def get_contract(name: str) -> SolverContract:
    """Return a contract by registry ID or exact native symbol.

    Raises:
        KeyError: If ``name`` is neither a contract ID nor a registered symbol.
    """

    if name in SOLVER_CONTRACTS:
        return SOLVER_CONTRACTS[name]
    if name in _BY_SYMBOL:
        return _BY_SYMBOL[name]
    raise KeyError(
        f"unknown solver contract {name!r}; use list_contracts() or "
        "list_native_solver_symbols() to discover valid names"
    )


def list_contracts() -> tuple[SolverContract, ...]:
    """Return every contract in stable registry order."""

    return _CONTRACTS


def list_native_solver_symbols() -> tuple[str, ...]:
    """Return all compiled entry-point names covered by the registry."""

    return tuple(sorted(_BY_SYMBOL))


def get_python_numerical_contract(name: str) -> PythonNumericalContract:
    """Return a Python numerical contract by registry ID or public symbol."""

    if name in PYTHON_NUMERICAL_CONTRACTS:
        return PYTHON_NUMERICAL_CONTRACTS[name]
    if name in _PYTHON_BY_SYMBOL:
        return _PYTHON_BY_SYMBOL[name]
    raise KeyError(
        f"unknown Python numerical contract {name!r}; use "
        "list_python_numerical_contracts() or list_python_numerical_symbols() "
        "to discover valid names"
    )


def list_python_numerical_contracts() -> tuple[PythonNumericalContract, ...]:
    """Return every public Python numerical/workflow contract in stable order."""

    return _PYTHON_NUMERICAL_CONTRACTS


def list_python_numerical_symbols() -> tuple[str, ...]:
    """Return public Python symbols owned by the separate numerical registry."""

    return tuple(sorted(_PYTHON_BY_SYMBOL))


def _contains_casefold(values: tuple[str, ...], expected: str) -> bool:
    target = expected.casefold()
    return any(value.casefold() == target for value in values)


def _coerce_evidence_level(
    value: Union[EvidenceLevel, str],
) -> EvidenceLevel:
    if isinstance(value, EvidenceLevel):
        return value
    try:
        return EvidenceLevel(value.casefold())
    except ValueError as error:
        choices = ", ".join(level.value for level in EvidenceLevel)
        raise ValueError(
            f"unknown evidence level {value!r}; choose one of {choices}"
        ) from error


def filter_contracts(
    *,
    dimension: Optional[str] = None,
    term: Optional[str] = None,
    boundary_condition: Optional[str] = None,
    minimum_evidence: Optional[Union[EvidenceLevel, str]] = None,
) -> tuple[SolverContract, ...]:
    """Filter contracts using exact, case-insensitive vocabulary values.

    ``minimum_evidence`` is a discovery convenience based on the strongest
    scoped record in a contract.  Callers should still inspect each record's
    claim before deciding whether it supports their intended inference.
    """

    threshold = (
        None if minimum_evidence is None else _coerce_evidence_level(minimum_evidence)
    )
    matches = []
    for contract in _CONTRACTS:
        if dimension is not None and not _contains_casefold(
            contract.supported_dimensions, dimension
        ):
            continue
        if term is not None and not _contains_casefold(contract.supported_terms, term):
            continue
        if boundary_condition is not None and not _contains_casefold(
            contract.supported_boundary_conditions, boundary_condition
        ):
            continue
        if (
            threshold is not None
            and _EVIDENCE_RANK[contract.evidence_level] < _EVIDENCE_RANK[threshold]
        ):
            continue
        matches.append(contract)
    return tuple(matches)


def registry_as_dict() -> dict[str, dict[str, object]]:
    """Return a JSON-serializable snapshot keyed by contract ID."""

    return {contract.contract_id: contract.to_dict() for contract in _CONTRACTS}


def python_registry_as_dict() -> dict[str, dict[str, object]]:
    """Return a JSON-ready snapshot of Python numerical/workflow contracts."""

    return {
        contract.contract_id: contract.to_dict()
        for contract in _PYTHON_NUMERICAL_CONTRACTS
    }


__all__ = [
    "EVIDENCE_DISCLAIMER",
    "EvidenceLevel",
    "EvidenceRecord",
    "PYTHON_NUMERICAL_CONTRACTS",
    "PythonBackend",
    "PythonNumericalContract",
    "SOLVER_CONTRACTS",
    "SolverContract",
    "filter_contracts",
    "get_contract",
    "get_python_numerical_contract",
    "list_contracts",
    "list_native_solver_symbols",
    "list_python_numerical_contracts",
    "list_python_numerical_symbols",
    "python_registry_as_dict",
    "registry_as_dict",
]
