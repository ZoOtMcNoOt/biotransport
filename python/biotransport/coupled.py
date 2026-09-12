"""Conservative, sparse transport networks with named domains and species.

All values are SI: metres, seconds, mol/m^3, m^3 and m^2. A model joins
well-mixed compartments and one-dimensional spatial domains through membranes.
Unconnected boundaries are sealed. Reactions act locally within each domain.
See ``docs/coupled_transport.md`` for equations, examples and numerical limits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence, Union

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import coo_matrix, csc_matrix

from ._core import Geometry, StructuredMesh
from .mesh_utils import _finite_float, _numeric_array, x_nodes
from .units import Dimension, Quantity

Rate = Callable[[float, Mapping[str, np.ndarray]], Union[float, np.ndarray]]
RateDerivative = Callable[[float, Mapping[str, np.ndarray]], Mapping[str, Any]]


def _name(value: str, kind: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(
            f"{kind} must be a nonempty name without surrounding whitespace"
        )
    return value


def _scalar(
    value: Any, name: str, *, positive: bool = False, dimension: Dimension | None = None
) -> float:
    if isinstance(value, Quantity) and dimension is not None:
        value = value.require(dimension)
    result = _finite_float(value, name)
    if result < 0.0 or (positive and result == 0.0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return result


def _owned(values: Any) -> np.ndarray:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _field(
    value: Any,
    size: int,
    name: str,
    *,
    nonnegative: bool = False,
    dimension: Dimension | None = None,
) -> np.ndarray:
    if isinstance(value, Quantity) and dimension is not None:
        value = value.require(dimension)
    result = _numeric_array(value, name)
    if result.ndim == 0:
        result = np.full(size, float(result))
    if result.shape != (size,):
        raise ValueError(
            f"{name} must be a scalar or have shape ({size},), got {result.shape}"
        )
    if nonnegative and np.any(result < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return _owned(result)


def _mapping(
    values: Mapping[str, Any] | None, species: tuple[str, ...], name: str
) -> Mapping[str, Any]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must map species names to values")
    unknown = set(values) - set(species)
    if unknown:
        raise ValueError(f"unknown species in {name}: {', '.join(map(str, unknown))}")
    return values


@dataclass(frozen=True)
class _Domain:
    name: str
    geometry: str
    x: np.ndarray
    volumes: np.ndarray
    face_areas: np.ndarray
    initial: np.ndarray
    diffusivity: np.ndarray

    @property
    def nodes(self) -> int:
        return self.volumes.size


@dataclass(frozen=True)
class _Membrane:
    name: str
    source: tuple[str, int]
    target: tuple[str, int]
    area: float
    permeability: np.ndarray
    partition: np.ndarray


@dataclass(frozen=True)
class _Reaction:
    domain: str
    stoichiometry: np.ndarray
    orders: np.ndarray | None = None
    constant: float = 0.0
    rate: Rate | None = None
    derivative: RateDerivative | None = None

    def evaluate(
        self,
        time: float,
        local: np.ndarray,
        species: tuple[str, ...],
        *,
        differentiate: bool = False,
    ) -> np.ndarray:
        if self.orders is not None:
            # Polynomial mass action also defines smooth Newton trial values.
            # Do not clip negatives: clipping would change both the law and Jacobian.
            if differentiate:
                result = np.zeros_like(local)
                for index in np.flatnonzero(self.orders):
                    powers = self.orders.copy()
                    powers[index] -= 1
                    result[index] = (
                        self.constant
                        * self.orders[index]
                        * np.prod(local ** powers[:, None], axis=0)
                    )
                return result
            return self.constant * np.prod(local ** self.orders[:, None], axis=0)

        # Callbacks receive owned read-only arrays, never views of integrator state.
        fields = MappingProxyType(
            {name: _owned(local[i]) for i, name in enumerate(species)}
        )
        if differentiate:
            assert self.derivative is not None
            returned = self.derivative(time, fields)
            if not isinstance(returned, Mapping):
                raise TypeError(
                    "reaction derivative must return a mapping of species to partial derivatives"
                )
            values = _mapping(returned, species, "reaction derivative")
            return np.vstack(
                [
                    _field(values.get(name, 0.0), local.shape[1], f"derivative[{name}]")
                    for name in species
                ]
            )
        assert self.rate is not None
        return _field(self.rate(time, fields), local.shape[1], "reaction rate")


class CoupledModel:
    """Build a transport network using physical names instead of state indices.

    Args:
        species: Unique names shared by every domain. Unspecified initial
            concentrations and diffusivities default to zero.

    The builder is mutable. ``compile()`` creates an independent numerical
    snapshot; later builder edits cannot change a compiled model or solution.
    """

    def __init__(self, species: Sequence[str]):
        if isinstance(species, (str, bytes)):
            raise TypeError("species must be a sequence of names, for example ['drug']")
        names = tuple(_name(value, "species") for value in species)
        if not names or len(set(names)) != len(names):
            raise ValueError(
                "species must contain at least one name, with no duplicates"
            )
        self._species = names
        self._domains: dict[str, _Domain] = {}
        self._membranes: dict[str, _Membrane] = {}
        self._reactions: list[_Reaction] = []
        self._invariants: dict[str, np.ndarray] = {}

    @property
    def species(self) -> tuple[str, ...]:
        """Names in the order used by the numerical state."""
        return self._species

    def _new_domain(self, name: str) -> str:
        name = _name(name, "domain")
        if name in self._domains:
            raise ValueError(f"domain {name!r} already exists")
        return name

    def _values(
        self,
        values: Mapping[str, Any] | None,
        size: int,
        label: str,
        dimension: Dimension,
    ) -> np.ndarray:
        values = _mapping(values, self.species, label)
        return _owned(
            np.vstack(
                [
                    _field(
                        values.get(name, 0.0),
                        size,
                        f"{label}[{name}]",
                        nonnegative=True,
                        dimension=dimension,
                    )
                    for name in self.species
                ]
            )
        )

    def compartment(
        self, name: str, *, volume: float, initial: Mapping[str, Any] | None = None
    ) -> CoupledModel:
        """Add a well-mixed compartment with volume in m^3 and initial mol/m^3."""
        name = self._new_domain(name)
        volume = _scalar(volume, "volume", positive=True)
        data = _Domain(
            name,
            "compartment",
            _owned([]),
            _owned([volume]),
            _owned([]),
            self._values(initial, 1, "initial", Dimension.MOLAR_CONCENTRATION),
            _owned(np.zeros((len(self.species), 1))),
        )
        self._domains[name] = data
        return self

    def domain(
        self,
        name: str,
        mesh: StructuredMesh,
        *,
        diffusivity: Mapping[str, Any] | None = None,
        initial: Mapping[str, Any] | None = None,
        cross_section: float | None = None,
        axial_length: float | Quantity | None = None,
    ) -> CoupledModel:
        """Add a 1D slab, radial cylinder or sphere with physical control volumes.

        A slab requires ``cross_section`` in m^2. A cylinder requires
        ``axial_length`` in metres. A sphere represents the full solid angle.
        Initial concentrations and diffusivities accept scalars or nodal arrays.
        Unconnected faces are sealed; radial symmetry follows from zero area.
        """
        name = self._new_domain(name)
        if not isinstance(mesh, StructuredMesh) or not mesh.is_1d():
            raise TypeError(
                "domain requires a 1D StructuredMesh, for example bt.mesh_1d(40)"
            )
        geometry = mesh.geometry()
        if geometry == Geometry.CARTESIAN:
            if cross_section is None or axial_length is not None:
                raise ValueError(
                    "a Cartesian domain requires cross_section and no axial_length"
                )
            factor = _scalar(cross_section, "cross_section", positive=True)
            label = "cartesian"
        elif geometry == Geometry.CYLINDRICAL:
            if axial_length is None or cross_section is not None:
                raise ValueError(
                    "a cylindrical domain requires axial_length and no cross_section"
                )
            factor = (
                2.0
                * math.pi
                * _scalar(
                    axial_length,
                    "axial_length",
                    positive=True,
                    dimension=Dimension.LENGTH,
                )
            )
            label = "cylindrical"
        else:
            if cross_section is not None or axial_length is not None:
                raise ValueError(
                    "a spherical domain takes neither cross_section nor axial_length"
                )
            factor, label = 4.0 * math.pi, "spherical"
        x = x_nodes(mesh)
        n = x.size
        volumes = _field(
            [factor * mesh.control_volume(i) for i in range(n)], n, "volumes"
        )
        areas = _field(
            [factor * mesh.lower_face_area(0)]
            + [factor * mesh.upper_face_area(i) for i in range(n)],
            n + 1,
            "face areas",
        )
        if np.any(volumes <= 0.0):
            raise ValueError(
                "physical control volumes must be positive and representable"
            )
        data = _Domain(
            name,
            label,
            _owned(x),
            volumes,
            areas,
            self._values(initial, n, "initial", Dimension.MOLAR_CONCENTRATION),
            self._values(diffusivity, n, "diffusivity", Dimension.DIFFUSIVITY),
        )
        self._domains[name] = data
        return self

    def _endpoint(self, endpoint: str | tuple[str, str]) -> tuple[str, int]:
        if isinstance(endpoint, str):
            domain = self._domains.get(endpoint)
            if domain is None:
                raise ValueError(
                    f"unknown domain {endpoint!r}; add it before connecting it"
                )
            if domain.geometry != "compartment":
                raise ValueError(
                    f"spatial endpoint needs a side: ({endpoint!r}, 'left' or 'right')"
                )
            return endpoint, 0
        if not isinstance(endpoint, tuple) or len(endpoint) != 2:
            raise TypeError(
                "an endpoint is a compartment name or (domain, 'left'/'right')"
            )
        name, side = endpoint
        if not isinstance(name, str) or name not in self._domains:
            raise ValueError(f"unknown domain {name!r}")
        domain = self._domains[name]
        if domain.geometry == "compartment" or side not in ("left", "right"):
            raise ValueError("only spatial domains have 'left' and 'right' endpoints")
        return name, 0 if side == "left" else domain.nodes - 1

    def membrane(
        self,
        name: str,
        source: str | tuple[str, str],
        target: str | tuple[str, str],
        *,
        area: float,
        permeability: float | Quantity | Mapping[str, Any],
        partition: float | Mapping[str, Any] = 1.0,
    ) -> CoupledModel:
        """Connect two endpoints with signed mol/s = P*A*(c_source - c_target/K).

        ``partition`` is K = target/source concentration at equilibrium.
        Permeability is in m/s. A scalar applies to all species; omitted mapping
        entries mean impermeable (P=0) or no partition preference (K=1).
        Interfaces sharing a spatial boundary may not exceed its physical area.
        """
        name = _name(name, "membrane")
        if name in self._membranes:
            raise ValueError(f"membrane {name!r} already exists")
        left, right = self._endpoint(source), self._endpoint(target)
        if left == right:
            raise ValueError("a membrane needs two different endpoints")
        area = _scalar(area, "area", positive=True)
        p = (
            _mapping(permeability, self.species, "permeability")
            if isinstance(permeability, Mapping)
            else dict.fromkeys(self.species, permeability)
        )
        k = (
            _mapping(partition, self.species, "partition")
            if isinstance(partition, Mapping)
            else dict.fromkeys(self.species, partition)
        )
        p_values = _owned(
            [
                _scalar(
                    p.get(s, 0.0),
                    f"permeability[{s}]",
                    dimension=Dimension.SOLUTE_PERMEABILITY,
                )
                for s in self.species
            ]
        )
        k_values = _owned(
            [
                _scalar(k.get(s, 1.0), f"partition[{s}]", positive=True)
                for s in self.species
            ]
        )
        for endpoint in (left, right):
            domain = self._domains[endpoint[0]]
            if domain.geometry != "compartment":
                available = domain.face_areas[0 if endpoint[1] == 0 else -1]
                used = sum(
                    m.area
                    for m in self._membranes.values()
                    if endpoint in (m.source, m.target)
                )
                if used + area > available * (1.0 + 1e-12):
                    raise ValueError(
                        f"membranes exceed physical area at {endpoint}; available {available:g} m^2"
                    )
        self._membranes[name] = _Membrane(name, left, right, area, p_values, k_values)
        return self

    def _stoichiometry(self, values: Mapping[str, Any], label: str) -> np.ndarray:
        values = _mapping(values, self.species, label)
        result = _owned(
            [_finite_float(values.get(s, 0.0), f"{label}[{s}]") for s in self.species]
        )
        if not np.any(result):
            raise ValueError(f"{label} must contain a nonzero coefficient")
        return result

    def _add_reaction(self, reaction: _Reaction) -> CoupledModel:
        if reaction.domain not in self._domains:
            raise ValueError(f"unknown reaction domain {reaction.domain!r}")
        for name, weights in self._invariants.items():
            self._check_invariant(name, weights, reaction)
        self._reactions.append(reaction)
        return self

    def mass_action(
        self,
        domain: str,
        *,
        reactants: Mapping[str, int],
        products: Mapping[str, float],
        rate_constant: float,
    ) -> CoupledModel:
        """Add a reaction with rate = k * product(c_species ** reactant_order).

        Reactant coefficients are positive integers and are also the kinetic
        orders. Product coefficients are positive numbers. Empty reactants or
        products represent production or loss. No factorial is included in k.
        Reversible chemistry uses two calls, one for each direction.
        """
        reactants = _mapping(reactants, self.species, "reactants")
        products = _mapping(products, self.species, "products")
        orders = np.zeros(len(self.species), dtype=int)
        net = np.zeros(len(self.species))
        for i, species in enumerate(self.species):
            if species in reactants:
                value = _scalar(
                    reactants[species], f"reactants[{species}]", positive=True
                )
                if not value.is_integer() or value > np.iinfo(np.int32).max:
                    raise ValueError(
                        "mass-action reactant orders must be positive integers"
                    )
                orders[i] = int(value)
            product = (
                _scalar(products[species], f"products[{species}]", positive=True)
                if species in products
                else 0.0
            )
            net[i] = product - orders[i]
        if not np.any(net):
            raise ValueError("reaction must have nonzero net stoichiometry")
        return self._add_reaction(
            _Reaction(
                domain,
                _owned(net),
                orders.copy(),
                _scalar(rate_constant, "rate_constant"),
            )
        )

    def reaction(
        self,
        domain: str,
        *,
        stoichiometry: Mapping[str, float],
        rate: Rate,
        derivative: RateDerivative,
    ) -> CoupledModel:
        """Add local custom kinetics and its analytic concentration derivatives.

        Callbacks receive ``(time, concentrations_by_species)`` and return a
        scalar or one value per domain node. ``derivative`` returns a mapping
        of species names to partial derivatives of the scalar rate. Missing
        derivatives mean zero. Keep callbacks pure; their closure state cannot
        be snapshotted. Rates are in mol/m^3/s and may depend explicitly on time.
        """
        if not callable(rate) or not callable(derivative):
            raise TypeError("custom reactions require callable rate and derivative")
        return self._add_reaction(
            _Reaction(
                domain,
                self._stoichiometry(stoichiometry, "stoichiometry"),
                rate=rate,
                derivative=derivative,
            )
        )

    @staticmethod
    def _check_invariant(name: str, weights: np.ndarray, reaction: _Reaction) -> None:
        # Normalize separately so even large, finite coefficients cannot turn
        # the conservation test into inf > inf (which silently passes).
        weights = weights / np.max(np.abs(weights))
        net = reaction.stoichiometry / np.max(np.abs(reaction.stoichiometry))
        scale = np.abs(weights) @ np.abs(net)
        if abs(weights @ net) > 32.0 * np.finfo(float).eps * scale:
            raise ValueError(
                f"reaction in {reaction.domain!r} violates conserved quantity {name!r}"
            )

    def conserve(self, name: str, weights: Mapping[str, float]) -> CoupledModel:
        """Declare and validate a conserved linear combination of species amounts.

        For A + B <-> C, weights {A: 1, C: 1} track the A moiety. Every reaction
        must preserve the declared combination; invalid edits are rejected.
        """
        name = _name(name, "conserved quantity")
        if name in self._invariants:
            raise ValueError(f"conserved quantity {name!r} already exists")
        values = self._stoichiometry(weights, "weights")
        for reaction in self._reactions:
            self._check_invariant(name, values, reaction)
        self._invariants[name] = values
        return self

    def compile(self) -> CompiledModel:
        """Freeze the configuration and assemble its sparse numerical operator."""
        if not self._domains:
            raise ValueError(
                "add at least one compartment or spatial domain before compiling"
            )
        return CompiledModel(self)

    def solve(
        self,
        end_time: float | Quantity,
        *,
        frames: int | None = None,
        save_at: Sequence[float] | None = None,
        method: str = "BDF",
        rtol: float = 1e-7,
        atol: float | Mapping[str, float] = 1e-10,
        max_step: float | None = None,
    ) -> CoupledSolution:
        """Compile and integrate; see :meth:`CompiledModel.solve` for controls."""
        return self.compile().solve(
            end_time,
            frames=frames,
            save_at=save_at,
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )


class CompiledModel:
    """An owned sparse model usable with BioTransport or external integrators.

    State order is domain insertion order, then species order, then node order.
    Use ``state_slice(domain, species)`` instead of manually computing indices.
    Linear mass-action kinetics are compiled once alongside diffusion and
    membranes; nonlinear reactions provide analytic sparse Jacobians.
    """

    def __init__(self, model: CoupledModel):
        self._species = model.species
        self._domains = dict(model._domains)
        self._membranes = dict(model._membranes)
        self._invariants = dict(model._invariants)
        self._slices: dict[tuple[str, str], slice] = {}
        self._domain_slices: dict[str, slice] = {}
        offset = 0
        for domain in self._domains.values():
            start = offset
            for species in self.species:
                self._slices[domain.name, species] = slice(
                    offset, offset + domain.nodes
                )
                offset += domain.nodes
            self._domain_slices[domain.name] = slice(start, offset)
        self._initial = _owned(
            np.concatenate([d.initial.ravel() for d in self._domains.values()])
        )
        self._volumes = _owned(
            np.concatenate(
                [np.tile(d.volumes, len(self.species)) for d in self._domains.values()]
            )
        )
        self._forcing = np.zeros(offset)
        rows: list[int] = []
        cols: list[int] = []
        values: list[float] = []

        def add(row, col, value):
            rows.extend(np.atleast_1d(row).tolist())
            cols.extend(np.atleast_1d(col).tolist())
            values.extend(np.atleast_1d(value).tolist())

        def transfer(left, right, conductance, partition=1.0):
            a = conductance / self._volumes[left]
            b = (conductance / partition) / self._volumes[right]
            add(left, left, -a)
            add(right, left, conductance / self._volumes[right])
            add(left, right, (conductance / partition) / self._volumes[left])
            add(right, right, -b)

        for domain in self._domains.values():
            if domain.nodes == 1:
                continue
            for s, species in enumerate(self.species):
                d1, d2 = domain.diffusivity[s, :-1], domain.diffusivity[s, 1:]
                lo, hi = np.minimum(d1, d2), np.maximum(d1, d2)
                # Harmonic mean without multiplying or summing large diffusivities.
                harmonic = 2.0 * (
                    lo / (1.0 + np.divide(lo, hi, out=np.zeros_like(lo), where=hi > 0))
                )
                g = harmonic * domain.face_areas[1:-1] / np.diff(domain.x)
                left = np.arange(
                    self.state_slice(domain.name, species).start,
                    self.state_slice(domain.name, species).stop - 1,
                )
                transfer(left, left + 1, g)
        for membrane in self._membranes.values():
            for i, species in enumerate(self.species):
                left = (
                    self.state_slice(membrane.source[0], species).start
                    + membrane.source[1]
                )
                right = (
                    self.state_slice(membrane.target[0], species).start
                    + membrane.target[1]
                )
                transfer(
                    left,
                    right,
                    membrane.area * membrane.permeability[i],
                    membrane.partition[i],
                )
        self._transport = coo_matrix(
            (values, (rows, cols)), shape=(offset, offset)
        ).tocsc()
        self._transport.eliminate_zeros()
        self._nonlinear: list[_Reaction] = []
        for reaction in model._reactions:
            domain = self._domains[reaction.domain]
            order = np.sum(reaction.orders) if reaction.orders is not None else -1
            if order == 0:
                for i, species in enumerate(self.species):
                    self._forcing[self.state_slice(domain.name, species)] += (
                        reaction.stoichiometry[i] * reaction.constant
                    )
            elif order == 1:
                assert reaction.orders is not None
                source_species = self.species[int(np.flatnonzero(reaction.orders)[0])]
                source = np.arange(
                    self.state_slice(domain.name, source_species).start,
                    self.state_slice(domain.name, source_species).stop,
                )
                for target_species_index in np.flatnonzero(reaction.stoichiometry):
                    target = np.arange(
                        self.state_slice(
                            domain.name, self.species[target_species_index]
                        ).start,
                        self.state_slice(
                            domain.name, self.species[target_species_index]
                        ).stop,
                    )
                    add(
                        target,
                        source,
                        np.full(
                            domain.nodes,
                            reaction.stoichiometry[target_species_index]
                            * reaction.constant,
                        ),
                    )
            else:
                self._nonlinear.append(reaction)
        self._linear = coo_matrix(
            (values, (rows, cols)), shape=(offset, offset)
        ).tocsc()
        self._linear.eliminate_zeros()
        if not np.all(np.isfinite(self._linear.data)) or not np.all(
            np.isfinite(self._forcing)
        ):
            raise ValueError("model coefficients overflow; rescale parameters or units")
        self._forcing = _owned(self._forcing)

    @property
    def species(self) -> tuple[str, ...]:
        """Species names in state order."""
        return self._species

    @property
    def domains(self) -> tuple[str, ...]:
        """Domain names in state order."""
        return tuple(self._domains)

    @property
    def initial_state(self) -> np.ndarray:
        """An owned flat concentration vector for external integrators."""
        return self._initial.copy()

    @property
    def transport_matrix(self) -> csc_matrix:
        """Sparse diffusion and membrane operator, excluding all reactions."""
        return self._transport.copy()

    @property
    def state_volumes(self) -> np.ndarray:
        """Physical m^3 for each concentration degree of freedom."""
        return self._volumes.copy()

    def state_slice(self, domain: str, species: str) -> slice:
        """Indices of one named concentration field in the flat state."""
        try:
            return self._slices[domain, species]
        except KeyError:
            raise ValueError(
                f"unknown domain/species pair: {domain!r}, {species!r}"
            ) from None

    def coordinates(self, domain: str) -> np.ndarray:
        """Spatial coordinates in metres; empty for a well-mixed compartment."""
        if domain not in self._domains:
            raise ValueError(f"unknown domain {domain!r}")
        return self._domains[domain].x.copy()

    def _state(self, state: np.ndarray) -> np.ndarray:
        result = _numeric_array(state, "state")
        if result.shape != self._initial.shape:
            raise ValueError(
                f"state must have shape {self._initial.shape}, got {result.shape}"
            )
        return result

    def _rhs(self, time: float, state: np.ndarray) -> np.ndarray:
        result = self._linear @ state + self._forcing
        for reaction in self._nonlinear:
            index = self._domain_slices[reaction.domain]
            local = state[index].reshape(len(self.species), -1)
            rate = reaction.evaluate(time, local, self.species)
            result[index] += (reaction.stoichiometry[:, None] * rate).ravel()
        if not np.all(np.isfinite(result)):
            raise ValueError("reaction or transport rate is nonfinite")
        return result

    def rhs(self, time: float, state: np.ndarray) -> np.ndarray:
        """Evaluate dc/dt in mol/m^3/s without changing the supplied state."""
        return self._rhs(_finite_float(time, "time"), self._state(state))

    def jacobian(self, time: float, state: np.ndarray) -> csc_matrix:
        """Analytic sparse derivative of rhs with respect to concentrations."""
        time = _finite_float(time, "time")
        state = self._state(state)
        rows, cols, values = [], [], []
        for reaction in self._nonlinear:
            index = self._domain_slices[reaction.domain]
            n = self._domains[reaction.domain].nodes
            derivatives = reaction.evaluate(
                time,
                state[index].reshape(len(self.species), n),
                self.species,
                differentiate=True,
            )
            nodes = np.arange(n)
            for i in np.flatnonzero(reaction.stoichiometry):
                for j in range(len(self.species)):
                    if np.any(derivatives[j]):
                        rows.extend((index.start + i * n + nodes).tolist())
                        cols.extend((index.start + j * n + nodes).tolist())
                        values.extend(
                            (reaction.stoichiometry[i] * derivatives[j]).tolist()
                        )
        result = (
            self._linear
            + coo_matrix((values, (rows, cols)), shape=self._linear.shape).tocsc()
        )
        if not np.all(np.isfinite(result.data)):
            raise ValueError("reaction Jacobian is nonfinite")
        return result

    def summary(self) -> str:
        """Describe state size, sparsity and domain geometry without solving."""
        parts = [
            f"CoupledModel: {len(self.domains)} domains, {len(self.species)} species, "
            f"{self._initial.size} concentrations",
            f"Species: {', '.join(self.species)}",
        ]
        parts.extend(
            f"  {d.name}: {d.geometry}, {d.nodes} nodes, volume={sum(d.volumes):.6g} m^3"
            for d in self._domains.values()
        )
        parts.append(
            f"{len(self._membranes)} membranes; {self._linear.nnz} linear nonzeros; "
            f"{len(self._nonlinear)} nonlinear reactions"
        )
        return "\n".join(parts)

    def solve(
        self,
        end_time: float | Quantity,
        *,
        frames: int | None = None,
        save_at: Sequence[float] | None = None,
        method: str = "BDF",
        rtol: float = 1e-7,
        atol: float | Mapping[str, float] = 1e-10,
        max_step: float | None = None,
    ) -> CoupledSolution:
        """Integrate all domains together with a sparse BDF or Radau solver.

        ``frames`` defaults to 40 output intervals; ``save_at`` specifies exact
        output times instead. Initial and final states are always included.
        Output sampling does not restart integration or set its internal steps.
        ``atol`` is in concentration units and may map every species to its own
        positive absolute tolerance. ``rtol`` is a positive relative tolerance.
        ``max_step`` can resolve fast explicit time dependence in custom rates.

        Adaptive implicit methods do not guarantee nonnegative concentrations.
        Results report the minimum without clipping. Check positivity and
        convergence at tighter tolerances before drawing scientific conclusions.
        """
        from .run import _saved_times

        final = _scalar(end_time, "end_time", dimension=Dimension.TIME)
        if method not in ("BDF", "Radau"):
            raise ValueError("method must be 'BDF' or 'Radau'")
        rtol = _scalar(rtol, "rtol", positive=True)
        if rtol < 100 * np.finfo(float).eps:
            raise ValueError("rtol must be at least 100 times machine epsilon")
        if isinstance(atol, Mapping):
            values = _mapping(atol, self.species, "atol")
            if set(values) != set(self.species):
                raise ValueError("atol must specify every species")
        else:
            values = dict.fromkeys(self.species, atol)
        tolerances = np.empty(self._initial.size)
        for (domain, species), index in self._slices.items():
            tolerances[index] = _scalar(
                values[species], f"atol[{species}]", positive=True
            )
        step = (
            np.inf
            if max_step is None
            else _scalar(max_step, "max_step", positive=True, dimension=Dimension.TIME)
        )
        # Validate sampling even for a zero-duration request.
        sample_final = final if final else 1.0
        requested = _saved_times(
            sample_final,
            None,
            save_at,
            frames if frames is not None else (40 if save_at is None else None),
        )
        if not final and save_at is not None and len(save_at):
            raise ValueError("save_at must be empty when end_time is zero")
        times = (
            np.asarray([0.0] + sorted(set(requested))) if final else np.asarray([0.0])
        )
        if final:
            jac = self.jacobian if self._nonlinear else self._linear
            result = solve_ivp(
                self._rhs,
                (0.0, final),
                self._initial.copy(),
                method=method,
                t_eval=times,
                jac=jac,
                rtol=rtol,
                atol=tolerances,
                max_step=step,
            )
            if not result.success or result.t[-1] != final:
                raise RuntimeError(f"coupled integration failed: {result.message}")
            states = result.y.T
            evaluations, jacobians, factorizations = (
                result.nfev,
                result.njev,
                result.nlu,
            )
        else:
            states = self._initial[None, :].copy()
            evaluations = jacobians = factorizations = 0
        if not np.all(np.isfinite(states)):
            raise RuntimeError("coupled integration returned nonfinite concentrations")
        diagnostics = CoupledDiagnostics(
            method,
            evaluations,
            jacobians,
            factorizations,
            float(np.min(states)),
            rtol,
            float(np.min(tolerances)),
            float(np.max(tolerances)),
        )
        return CoupledSolution(self, times, states, diagnostics)


@dataclass(frozen=True)
class CoupledDiagnostics:
    """Observed integrator work and concentration range at saved times.

    Output frame count is not an internal step count. ``minimum_concentration``
    covers saved states only. Relative and absolute tolerances control local
    error estimates, not a guarantee on global solution error.
    """

    method: str
    rhs_evaluations: int
    jacobian_evaluations: int
    factorizations: int
    minimum_concentration: float
    rtol: float
    minimum_atol: float
    maximum_atol: float


@dataclass(frozen=True)
class ConservationReport:
    """Drift of a declared conserved amount over saved states.

    ``relative_drift`` is undefined (None) when the initial conserved amount is
    zero. Absolute drift is always available in weighted moles.
    """

    name: str
    initial: float
    final: float
    maximum_absolute_drift: float
    relative_drift: float | None


class CoupledSolution:
    """Owned saved fields with named access, membrane rates and amount balances."""

    def __init__(
        self,
        model: CompiledModel,
        times: np.ndarray,
        states: np.ndarray,
        diagnostics: CoupledDiagnostics,
    ):
        self._model = model
        self._times = _owned(times)
        self._states = _owned(states)
        self._diagnostics = diagnostics

    @property
    def times(self) -> np.ndarray:
        """Saved physical times in seconds, including zero and end_time."""
        return self._times.copy()

    @property
    def diagnostics(self) -> CoupledDiagnostics:
        """Immutable integration diagnostics."""
        return self._diagnostics

    def history(self, domain: str, species: str) -> np.ndarray:
        """Concentrations with shape (saved times, domain nodes), as an owned array."""
        return self._states[:, self._model.state_slice(domain, species)].copy()

    def field(self, domain: str, species: str) -> np.ndarray:
        """Final concentration field, including a length-one field for compartments."""
        return self._states[-1, self._model.state_slice(domain, species)].copy()

    def amount(self, species: str, domain: str | None = None) -> np.ndarray:
        """Moles at every saved time, summed across all domains by default."""
        domains = self._model.domains if domain is None else (domain,)
        result = np.zeros(self._times.size)
        for name in domains:
            index = self._model.state_slice(name, species)
            result += self._states[:, index] @ self._model._domains[name].volumes
        if not np.all(np.isfinite(result)):
            raise ValueError(
                "species amount overflows; rescale concentrations or volumes"
            )
        return result

    def interface_rate(self, name: str, species: str) -> np.ndarray:
        """Signed source-to-target membrane transport rate in mol/s at saved times."""
        if name not in self._model._membranes:
            raise ValueError(f"unknown membrane {name!r}")
        membrane = self._model._membranes[name]
        source = self.history(membrane.source[0], species)[:, membrane.source[1]]
        target = self.history(membrane.target[0], species)[:, membrane.target[1]]
        i = self._model.species.index(species)
        return (
            membrane.area
            * membrane.permeability[i]
            * (source - target / membrane.partition[i])
        )

    def balance(self, name: str) -> ConservationReport:
        """Check a quantity previously declared with ``model.conserve``."""
        if name not in self._model._invariants:
            raise ValueError(
                f"unknown conserved quantity {name!r}; declare it with model.conserve"
            )
        weights = self._model._invariants[name]
        amounts = sum(
            weights[i] * self.amount(s) for i, s in enumerate(self._model.species)
        )
        if not np.all(np.isfinite(amounts)):
            raise ValueError("conserved amount overflows; rescale conservation weights")
        initial, final = float(amounts[0]), float(amounts[-1])
        drift = float(np.max(np.abs(amounts - initial)))
        return ConservationReport(
            name, initial, final, drift, drift / abs(initial) if initial else None
        )
