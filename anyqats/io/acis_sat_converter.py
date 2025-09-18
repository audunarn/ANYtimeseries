"""Utility for converting ACIS SAT files.

This module implements a converter that parses an ACIS SAT file and exposes
its contents both in textual representations (JSON, CSV, Markdown, pandas
``DataFrame``) and in geometric form.  The geometry pipeline performs a best
-effort reconstruction of the boundary representation contained in the SAT
model by analysing the entity graph (faces, loops, coedges, edges, vertices and
points).  The reconstructed mesh can be exported to commonly used formats such
as ``.brep`` (a compact JSON based boundary representation used by
:mod:`anyqats`), ``.obj`` and ``.stl``.

Example
-------

>>> from anyqats.io.acis_sat_converter import ACISSATConverter
>>> converter = ACISSATConverter("model.sat")
>>> mesh = converter.to_mesh()  # doctest: +SKIP
>>> converter.to_brep("model.brep")  # doctest: +SKIP

The resulting ``model.brep`` file contains a triangulated boundary
representation that can be consumed by downstream tooling.
"""

from __future__ import annotations

import csv
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

import pandas as pd

__all__ = ["SATEntity", "BRepMesh", "ACISSATConverter"]


@dataclass(frozen=True)
class SATEntity:
    """Representation of a single SAT entity.

    Parameters
    ----------
    entity_type:
        Name of the entity – e.g. ``"body"``, ``"face"`` or ``"edge"``.
    identifier:
        Identifier of the entity if present (ACIS typically denotes them with
        the ``$`` prefix).  ``None`` if the entity line does not contain an
        explicit identifier.
    tokens:
        Remaining tokens on the entity line.  The converter performs a light
        weight normalisation by converting numeric values to ``int``/``float``
        instances when possible while keeping symbolic values untouched.
    record_index:
        Optional numeric index that prefixes the entity definition in ACIS
        files.  Older SAT exports omit this counter whereas newer variants use
        a signed integer (e.g. ``-42``).  The converter keeps the value for
        reference but it is not required for downstream processing.
    reference_key:
        Canonical identifier used internally by :class:`ACISSATConverter` when
        resolving references between entities.  This corresponds to either the
        explicit SAT identifier (e.g. ``"$42"``) or a synthetic key derived
        from the record index (``"@42"``) when the SAT export omits explicit
        identifiers.
    """

    entity_type: str
    identifier: Optional[str]
    tokens: List[Union[str, int, float]]
    record_index: Optional[int] = None
    reference_key: Optional[str] = None

    def to_dict(self) -> Dict[str, Union[str, int, float, List[Union[str, int, float]]]]:
        """Return a dictionary representation suitable for serialisation."""

        return {
            "entity_type": self.entity_type,
            "identifier": self.identifier,
            "tokens": self.tokens,
            "record_index": self.record_index,
            "reference_key": self.reference_key,
        }


@dataclass(frozen=True)
class BRepMesh:
    """Triangulated boundary representation extracted from a SAT model."""

    vertices: List[Tuple[float, float, float]]
    faces: List[Tuple[int, int, int]]

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the mesh to a JSON friendly dictionary."""

        return {
            "vertices": [list(map(float, vertex)) for vertex in self.vertices],
            "faces": [list(map(int, face)) for face in self.faces],
        }

    @property
    def is_empty(self) -> bool:
        """Return ``True`` when the mesh does not contain any faces."""

        return not self.vertices or not self.faces


@dataclass(frozen=True)
class _CoedgeData:
    """Internal container describing the topology of a coedge."""

    edge: Optional[str]
    next: Optional[str]
    previous: Optional[str]
    partner: Optional[str]
    orientation: str


class ACISSATConverter:
    """Parse an ACIS SAT file and convert it to other formats."""

    #: Default comment markers that are stripped from SAT lines.
    DEFAULT_COMMENT_PREFIXES: Sequence[str] = ("//", "--", "!")

    def __init__(
        self,
        sat_path: Union[str, Path],
        *,
        encoding: str = "utf-8",
        comment_prefixes: Optional[Sequence[str]] = None,
    ) -> None:
        self.sat_path = Path(sat_path)
        self.encoding = encoding
        if comment_prefixes is None:
            self.comment_prefixes = tuple(self.DEFAULT_COMMENT_PREFIXES)
        else:
            if not comment_prefixes:
                raise ValueError("comment_prefixes must not be empty")
            self.comment_prefixes = tuple(comment_prefixes)

        self._header: Dict[str, Any] = {}
        self._entities: List[SATEntity] = []
        self._entity_index: Dict[str, SATEntity] = {}
        self._is_parsed: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def header(self) -> Dict[str, Any]:
        """Return the header metadata extracted from the SAT file."""

        self._ensure_parsed()
        return self._header

    @property
    def entities(self) -> List[SATEntity]:
        """List of :class:`SATEntity` parsed from the SAT file."""

        self._ensure_parsed()
        return self._entities

    def to_mesh(self) -> BRepMesh:
        """Return the triangulated boundary representation of the SAT model.

        The mesh reconstruction is a best-effort procedure that currently
        supports planar faces bounded by loops of straight edges.  When the SAT
        model contains more advanced geometry (e.g. splines, cylinders or
        non-manifold edges) the converter will raise a :class:`ValueError`.
        """

        self._ensure_parsed()
        builder = _SATGeometryBuilder(self._entities, self._entity_index)
        return builder.build_mesh()

    # Conversion helpers -------------------------------------------------
    def convert(self, destination: Union[str, Path], format: Optional[str] = None, **kwargs) -> Path:
        """Convert the SAT content to another format.

        Parameters
        ----------
        destination:
            Target file that should receive the converted data.
        format:
            Optional string describing the desired output format.  When not
            provided the converter infers the format from the file suffix of
            ``destination``.
        **kwargs:
            Forwarded to the specific conversion method.

        Returns
        -------
        pathlib.Path
            Path to the created file.
        """

        target = Path(destination)
        fmt = (format or target.suffix.lstrip(".")).lower()
        if not fmt:
            raise ValueError("Unable to determine output format; please provide a format explicitly")

        if fmt == "json":
            self.to_json(target, **kwargs)
        elif fmt in {"csv", "tsv"}:
            delimiter = "," if fmt == "csv" else "\t"
            if "delimiter" in kwargs:
                delimiter = kwargs["delimiter"]
            self.to_csv(target, delimiter=delimiter)
        elif fmt in {"md", "markdown"}:
            self.to_markdown(target)
        elif fmt == "brep":
            self.to_brep(target)
        elif fmt == "obj":
            self.to_obj(target)
        elif fmt == "stl":
            self.to_stl(target)
        else:
            raise ValueError(f"Unsupported conversion format: {fmt}")

        return target

    def as_dict(self) -> Dict[str, Any]:
        """Return the parsed SAT file as a dictionary structure."""

        data = {
            "source": str(self.sat_path),
            "header": self.header,
            "entities": [entity.to_dict() for entity in self.entities],
            "entity_counts": self.entity_counts(),
        }
        return data

    def entity_counts(self) -> Dict[str, int]:
        """Return a dictionary with the number of occurrences per entity type."""

        self._ensure_parsed()
        counts: Dict[str, int] = {}
        for entity in self._entities:
            counts[entity.entity_type] = counts.get(entity.entity_type, 0) + 1
        return counts

    def to_json(self, destination: Union[str, Path], *, indent: int = 2) -> None:
        """Write the parsed SAT data to a JSON file."""

        target = Path(destination)
        data = self.as_dict()
        with target.open("w", encoding=self.encoding) as json_file:
            json.dump(data, json_file, indent=indent)

    def to_csv(self, destination: Union[str, Path], *, delimiter: str = ",") -> None:
        """Write the parsed SAT entities to a delimited file."""

        target = Path(destination)
        with target.open("w", newline="", encoding=self.encoding) as csv_file:
            writer = csv.writer(csv_file, delimiter=delimiter)
            writer.writerow(["entity_type", "identifier", "tokens"])
            for entity in self.entities:
                token_string = " ".join(str(value) for value in entity.tokens)
                writer.writerow([entity.entity_type, entity.identifier or "", token_string])

    def to_markdown(self, destination: Union[str, Path]) -> None:
        """Serialise a quick inspection table in GitHub flavoured markdown."""

        target = Path(destination)
        header = "| entity_type | identifier | token_count |\n"
        separator = "|---|---|---|\n"
        rows = [
            f"| {entity.entity_type} | {entity.identifier or ''} | {len(entity.tokens)} |"
            for entity in self.entities
        ]
        body = "\n".join(rows)
        table = header + separator
        if body:
            table += body + "\n"
        target.write_text(table, encoding=self.encoding)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the parsed entities as a :class:`pandas.DataFrame`."""

        records = [
            {
                "entity_type": entity.entity_type,
                "identifier": entity.identifier,
                "tokens": entity.tokens,
            }
            for entity in self.entities
        ]
        return pd.DataFrame.from_records(records)

    def to_brep(self, destination: Union[str, Path]) -> None:
        """Export the reconstructed mesh as an ``.brep`` file.

        The ``.brep`` format used by :mod:`anyqats` is a compact JSON based
        representation that stores all vertices and triangular faces.  The
        resulting file can be read with the :meth:`to_mesh` method.
        """

        target = Path(destination)
        mesh = self.to_mesh()
        if mesh.is_empty:
            raise ValueError("The SAT model does not contain triangulable faces")

        payload = {
            "format": "anyqats-brep",
            "version": "1.0",
            "source": str(self.sat_path),
            "mesh": mesh.to_dict(),
        }
        target.write_text(json.dumps(payload, indent=2), encoding=self.encoding)

    def to_obj(self, destination: Union[str, Path]) -> None:
        """Export the reconstructed mesh to the Wavefront ``.obj`` format."""

        target = Path(destination)
        mesh = self.to_mesh()
        if mesh.is_empty:
            raise ValueError("The SAT model does not contain triangulable faces")

        with target.open("w", encoding=self.encoding) as obj_file:
            obj_file.write(f"# Generated from {self.sat_path}\n")
            for vertex in mesh.vertices:
                obj_file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
            for face in mesh.faces:
                # OBJ uses 1-based indexing
                obj_file.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    def to_stl(self, destination: Union[str, Path]) -> None:
        """Export the reconstructed mesh to an ASCII ``.stl`` file."""

        target = Path(destination)
        mesh = self.to_mesh()
        if mesh.is_empty:
            raise ValueError("The SAT model does not contain triangulable faces")

        with target.open("w", encoding=self.encoding) as stl_file:
            stl_file.write(f"solid sat_model\n")
            for face in mesh.faces:
                p0 = mesh.vertices[face[0]]
                p1 = mesh.vertices[face[1]]
                p2 = mesh.vertices[face[2]]
                normal = _triangle_normal(p0, p1, p2)
                stl_file.write(
                    f"  facet normal {normal[0]} {normal[1]} {normal[2]}\n"
                    "    outer loop\n"
                    f"      vertex {p0[0]} {p0[1]} {p0[2]}\n"
                    f"      vertex {p1[0]} {p1[1]} {p1[2]}\n"
                    f"      vertex {p2[0]} {p2[1]} {p2[2]}\n"
                    "    endloop\n"
                    "  endfacet\n"
                )
            stl_file.write("endsolid sat_model\n")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_parsed(self) -> None:
        if not self._is_parsed:
            self._parse()

    def _parse(self) -> None:
        if not self.sat_path.exists():
            raise FileNotFoundError(f"SAT file not found: {self.sat_path}")

        raw_lines = self.sat_path.read_text(encoding=self.encoding).splitlines()
        header_lines, entity_lines = self._split_header_and_body(raw_lines)
        self._header = self._parse_header(header_lines)
        self._entities = list(self._parse_entities(entity_lines))
        self._entity_index = {}
        for entity in self._entities:
            for key in self._reference_keys_for_entity(entity):
                if key in self._entity_index:
                    continue
                self._entity_index[key] = entity
        self._is_parsed = True

    def _split_header_and_body(self, lines: Iterable[str]) -> Tuple[List[str], List[str]]:
        header_lines: List[str] = []
        entity_lines: List[str] = []
        in_body = False

        for raw_line in lines:
            stripped = raw_line.strip()
            if not stripped:
                continue

            cleaned_with_hash = self._strip_comments(stripped, keep_hash=True)
            if not cleaned_with_hash:
                continue

            if not in_body:
                if "#" in cleaned_with_hash:
                    in_body = True
                else:
                    header_lines.append(self._strip_comments(stripped, keep_hash=False))
                    continue

            entity_lines.append(cleaned_with_hash)

        return header_lines, entity_lines

    def _parse_header(self, header_lines: Sequence[str]) -> Dict[str, Any]:
        header: Dict[str, Any] = {"raw_lines": list(header_lines)}

        if header_lines:
            header["tokenised_lines"] = [self._tokenise_and_normalise(line) for line in header_lines]

            version_tokens = header["tokenised_lines"][0]
            numeric_tokens = [token for token in version_tokens if isinstance(token, (int, float))]
            if numeric_tokens:
                header["version_numbers"] = numeric_tokens

        if len(header_lines) > 1:
            header["metadata"] = header_lines[1:]

        return header

    def _parse_entities(self, lines: Iterable[str]) -> Iterator[SATEntity]:
        for line in lines:
            tokens = self._remove_entity_terminator(line.split())
            if not tokens:
                continue

            record_index, tokens = self._extract_leading_index(tokens)
            if not tokens:
                continue

            entity_type = tokens[0]
            remainder = tokens[1:]
            identifier: Optional[str] = None
            if remainder and isinstance(remainder[0], str) and remainder[0].startswith("$"):
                identifier = remainder[0]
                remainder = remainder[1:]

            normalised_tokens = [self._normalise_token(token) for token in remainder]
            reference_key = self._determine_reference_key(record_index, identifier)
            yield SATEntity(
                entity_type=entity_type,
                identifier=identifier,
                tokens=normalised_tokens,
                record_index=record_index,
                reference_key=reference_key,
            )

    # Normalisation utilities ------------------------------------------
    def _determine_reference_key(
        self, record_index: Optional[int], identifier: Optional[str]
    ) -> Optional[str]:
        if identifier and self._should_index_identifier(identifier):
            return identifier
        if record_index is not None:
            return f"@{abs(record_index)}"
        return None

    def _reference_keys_for_entity(self, entity: SATEntity) -> Iterator[str]:
        seen: set[str] = set()
        if entity.reference_key:
            seen.add(entity.reference_key)
            yield entity.reference_key
        if entity.record_index is not None:
            key = f"@{abs(entity.record_index)}"
            if key not in seen:
                seen.add(key)
                yield key

    def _should_index_identifier(self, identifier: str) -> bool:
        if not identifier.startswith("$"):
            return True
        try:
            return int(identifier[1:]) >= 0
        except ValueError:
            return True

    def _strip_comments(self, line: str, *, keep_hash: bool) -> str:
        cleaned = line
        for prefix in self.comment_prefixes:
            idx = cleaned.find(prefix)
            if idx != -1:
                cleaned = cleaned[:idx]
        cleaned = cleaned.rstrip()
        if not keep_hash:
            hash_idx = cleaned.find("#")
            if hash_idx != -1:
                cleaned = cleaned[:hash_idx]
        return cleaned.strip()

    def _remove_entity_terminator(self, tokens: List[str]) -> List[str]:
        cleaned: List[str] = []
        for token in tokens:
            if token == "#":
                break
            if token.endswith("#"):
                token = token[:-1]
                if token:
                    cleaned.append(token)
                break
            cleaned.append(token)
        return cleaned

    def _extract_leading_index(self, tokens: List[str]) -> Tuple[Optional[int], List[str]]:
        if not tokens:
            return None, tokens
        first = tokens[0]
        if self._is_integer_token(first):
            try:
                return int(first), tokens[1:]
            except ValueError:
                return None, tokens[1:]
        return None, tokens

    def _is_integer_token(self, token: str) -> bool:
        if not token:
            return False
        if token[0] in {"+", "-"}:
            token = token[1:]
        return token.isdigit()

    def _normalise_token(self, token: str) -> Union[str, int, float]:
        if not token:
            return token
        if token.startswith("$"):
            return token

        return self._convert_numeric_token(token)

    def _tokenise_and_normalise(self, line: str) -> List[Union[str, int, float]]:
        return [self._normalise_token(token) for token in line.split()]

    def _convert_numeric_token(self, token: str) -> Union[str, int, float]:
        try:
            return int(token)
        except ValueError:
            try:
                return float(token)
            except ValueError:
                return token


class _SATGeometryBuilder:
    """Internal helper that converts SAT topology to a :class:`BRepMesh`."""

    _EPS = 1e-9

    def __init__(self, entities: Sequence[SATEntity], entity_index: Dict[str, SATEntity]):
        self.entities = entities
        self.entity_index = entity_index
        self.points: Dict[str, Tuple[float, float, float]] = {}
        self.vertices: Dict[str, Tuple[float, float, float]] = {}
        self.edges: Dict[str, Tuple[str, str]] = {}
        self.coedges: Dict[str, _CoedgeData] = {}
        self.loops: Dict[str, List[str]] = {}
        self.faces: Dict[str, List[str]] = {}

    # Public API -------------------------------------------------------
    def build_mesh(self) -> BRepMesh:
        self._collect_points()
        self._collect_vertices()
        self._collect_edges()
        self._collect_coedges()
        self._collect_loops()
        self._collect_faces()
        vertices, faces = self._build_faces()
        if not faces:
            raise ValueError("Unable to reconstruct a triangulated mesh from the SAT model")
        return BRepMesh(vertices=vertices, faces=faces)

    # Collection helpers ----------------------------------------------
    def _collect_points(self) -> None:
        for entity in self.entities:
            if entity.entity_type != "point" or entity.reference_key is None:
                continue
            coords = self._extract_coordinates(entity.tokens)
            if coords is None:
                warnings.warn(
                    f"Point {entity.reference_key} does not contain three numeric coordinates; skipping",
                    RuntimeWarning,
                )
                continue
            self.points[entity.reference_key] = coords

    def _collect_vertices(self) -> None:
        for entity in self.entities:
            if not entity.entity_type.startswith("vertex") or entity.reference_key is None:
                continue
            point_ref = self._find_first_reference(entity.tokens, {"point"})
            if not point_ref:
                warnings.warn(
                    f"Vertex {entity.reference_key} does not reference a point entity; skipping",
                    RuntimeWarning,
                )
                continue
            point = self.points.get(point_ref)
            if point is None:
                warnings.warn(
                    f"Point {point_ref} referenced by vertex {entity.reference_key} is missing; skipping",
                    RuntimeWarning,
                )
                continue
            self.vertices[entity.reference_key] = point

    def _collect_edges(self) -> None:
        for entity in self.entities:
            if not entity.entity_type.startswith("edge") or entity.reference_key is None:
                continue
            vertex_refs = self._find_all_references(entity.tokens, {"vertex"})
            unique_vertices: List[str] = []
            for ref in vertex_refs:
                if ref not in unique_vertices:
                    unique_vertices.append(ref)
            if len(unique_vertices) < 2:
                warnings.warn(
                    f"Edge {entity.reference_key} does not reference two vertices; skipping",
                    RuntimeWarning,
                )
                continue
            self.edges[entity.reference_key] = (unique_vertices[0], unique_vertices[1])

    def _collect_coedges(self) -> None:
        for entity in self.entities:
            if not entity.entity_type.startswith("coedge") or entity.reference_key is None:
                continue
            orientation = "reversed" if self._has_token(entity.tokens, "reversed") else "forward"
            adjacency: List[str] = []
            edge_ref: Optional[str] = None
            for token in entity.tokens:
                key = self._resolve_reference_key(token)
                if key is None:
                    continue
                referenced = self.entity_index.get(key)
                if referenced is None:
                    continue
                if referenced.entity_type.startswith("coedge") and len(adjacency) < 3:
                    adjacency.append(key)
                elif referenced.entity_type.startswith("edge") and edge_ref is None:
                    edge_ref = key
                if edge_ref is not None and len(adjacency) >= 3:
                    break
            if edge_ref is None:
                warnings.warn(
                    f"Coedge {entity.reference_key} does not reference an edge; skipping",
                    RuntimeWarning,
                )
                continue
            next_ref = adjacency[0] if len(adjacency) > 0 else None
            previous_ref = adjacency[1] if len(adjacency) > 1 else None
            partner_ref = adjacency[2] if len(adjacency) > 2 else None
            self.coedges[entity.reference_key] = _CoedgeData(
                edge=edge_ref,
                next=next_ref,
                previous=previous_ref,
                partner=partner_ref,
                orientation=orientation,
            )

    def _collect_loops(self) -> None:
        for entity in self.entities:
            if not entity.entity_type.startswith("loop") or entity.reference_key is None:
                continue
            start_coedge = self._find_first_reference(entity.tokens, {"coedge"})
            if not start_coedge:
                warnings.warn(
                    f"Loop {entity.reference_key} does not reference any coedges; skipping",
                    RuntimeWarning,
                )
                continue
            coedge_cycle = self._traverse_coedge_cycle(start_coedge, entity.reference_key)
            if not coedge_cycle:
                continue
            self.loops[entity.reference_key] = coedge_cycle

    def _traverse_coedge_cycle(self, start_coedge: str, loop_id: str) -> List[str]:
        if start_coedge not in self.coedges:
            warnings.warn(
                f"Loop {loop_id} references missing coedge {start_coedge}; skipping",
                RuntimeWarning,
            )
            return []

        visited: Set[str] = set()
        sequence: List[str] = []
        current = start_coedge

        while current not in visited:
            visited.add(current)
            sequence.append(current)
            data = self.coedges.get(current)
            if data is None:
                warnings.warn(
                    f"Loop {loop_id} encountered missing coedge {current}; skipping",
                    RuntimeWarning,
                )
                return []
            next_coedge = data.next
            if next_coedge is None:
                warnings.warn(
                    f"Coedge {current} on loop {loop_id} does not specify a next pointer; skipping loop",
                    RuntimeWarning,
                )
                return []
            if next_coedge not in self.coedges:
                warnings.warn(
                    f"Coedge {current} on loop {loop_id} references unknown next coedge {next_coedge}; skipping loop",
                    RuntimeWarning,
                )
                return []
            current = next_coedge

        if current != start_coedge:
            warnings.warn(
                f"Loop {loop_id} traversal terminated at {current} without returning to {start_coedge}; skipping",
                RuntimeWarning,
            )
            return []

        return sequence

    def _collect_faces(self) -> None:
        for entity in self.entities:
            if not entity.entity_type.startswith("face") or entity.reference_key is None:
                continue
            loop_refs = self._find_all_references(entity.tokens, {"loop"})
            if not loop_refs:
                warnings.warn(
                    f"Face {entity.reference_key} does not contain any loops; skipping",
                    RuntimeWarning,
                )
                continue
            self.faces[entity.reference_key] = loop_refs

    # Mesh construction -----------------------------------------------
    def _build_faces(self) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
        vertex_index: Dict[str, int] = {}
        mesh_vertices: List[Tuple[float, float, float]] = []
        mesh_faces: List[Tuple[int, int, int]] = []

        for face_id, loop_ids in self.faces.items():
            if not loop_ids:
                continue
            if len(loop_ids) > 1:
                warnings.warn(
                    f"Face {face_id} contains {len(loop_ids) - 1} inner loop(s); holes are currently ignored",
                    RuntimeWarning,
                )
            outer_loop_id = loop_ids[0]
            loop_vertex_ids = self._build_loop_vertex_sequence(outer_loop_id)
            if len(loop_vertex_ids) < 3:
                warnings.warn(
                    f"Loop {outer_loop_id} on face {face_id} could not be resolved; skipping face",
                    RuntimeWarning,
                )
                continue
            try:
                coordinates = [self.vertices[vertex_id] for vertex_id in loop_vertex_ids]
            except KeyError as exc:
                warnings.warn(
                    f"Missing vertex {exc.args[0]} while processing face {face_id}; skipping face",
                    RuntimeWarning,
                )
                continue

            try:
                triangles = self._triangulate_polygon(coordinates)
            except ValueError as exc:
                warnings.warn(
                    f"Face {face_id} could not be triangulated ({exc}); skipping",
                    RuntimeWarning,
                )
                continue

            for tri in triangles:
                global_indices: List[int] = []
                for local_index in tri:
                    vertex_id = loop_vertex_ids[local_index]
                    global_idx = vertex_index.get(vertex_id)
                    if global_idx is None:
                        global_idx = len(mesh_vertices)
                        mesh_vertices.append(coordinates[local_index])
                        vertex_index[vertex_id] = global_idx
                    global_indices.append(global_idx)
                mesh_faces.append(tuple(global_indices))

        return mesh_vertices, mesh_faces

    # Utility helpers --------------------------------------------------
    def _extract_coordinates(self, tokens: Sequence[Union[str, int, float]]) -> Optional[Tuple[float, float, float]]:
        numeric_tokens = [float(token) for token in tokens if isinstance(token, (int, float))]
        if len(numeric_tokens) < 3:
            return None
        return tuple(numeric_tokens[-3:])  # type: ignore[return-value]

    def _find_first_reference(self, tokens: Sequence[Union[str, int, float]], prefixes: Sequence[str]) -> Optional[str]:
        for token in tokens:
            key = self._resolve_reference_key(token)
            if key is None:
                continue
            referenced = self.entity_index.get(key)
            if referenced and any(referenced.entity_type.startswith(prefix) for prefix in prefixes):
                return key
        return None

    def _find_all_references(self, tokens: Sequence[Union[str, int, float]], prefixes: Sequence[str]) -> List[str]:
        references: List[str] = []
        for token in tokens:
            key = self._resolve_reference_key(token)
            if key is None:
                continue
            referenced = self.entity_index.get(key)
            if referenced and any(referenced.entity_type.startswith(prefix) for prefix in prefixes):
                references.append(key)
        return references

    def _has_token(self, tokens: Sequence[Union[str, int, float]], value: str) -> bool:
        return any(isinstance(token, str) and token.lower() == value for token in tokens)

    def _resolve_reference_key(self, token: Union[str, int, float]) -> Optional[str]:
        if not isinstance(token, str) or not token.startswith("$"):
            return None
        direct = self.entity_index.get(token)
        if direct is not None:
            return direct.reference_key or token
        suffix = token[1:]
        if suffix.isdigit():
            fallback_key = f"@{suffix}"
            fallback = self.entity_index.get(fallback_key)
            if fallback is not None:
                return fallback.reference_key or fallback_key
        return None

    def _build_loop_vertex_sequence(self, loop_id: str) -> List[str]:
        coedge_ids = self.loops.get(loop_id, [])
        edges: List[Tuple[str, str]] = []
        for coedge_id in coedge_ids:
            coedge_data = self.coedges.get(coedge_id)
            if coedge_data is None:
                warnings.warn(
                    f"Loop {loop_id} references missing coedge {coedge_id}; skipping",
                    RuntimeWarning,
                )
                continue
            edge_id = coedge_data.edge
            if edge_id is None:
                warnings.warn(
                    f"Coedge {coedge_id} on loop {loop_id} does not reference an edge; skipping",
                    RuntimeWarning,
                )
                continue
            vertex_pair = self.edges.get(edge_id)
            if not vertex_pair:
                warnings.warn(
                    f"Edge {edge_id} referenced by coedge {coedge_id} is missing; skipping",
                    RuntimeWarning,
                )
                continue
            start, end = vertex_pair
            if coedge_data.orientation.lower() == "reversed":
                start, end = end, start
            edges.append((start, end))

        if not edges:
            return []

        try:
            ordered_edges = self._order_edges(edges)
        except ValueError as exc:
            warnings.warn(f"Loop {loop_id} could not be ordered ({exc})", RuntimeWarning)
            return []

        vertex_ids: List[str] = [ordered_edges[0][0]]
        for _, end in ordered_edges:
            if end == vertex_ids[0] and len(vertex_ids) > 2:
                continue
            vertex_ids.append(end)
        if len(vertex_ids) > 1 and vertex_ids[0] == vertex_ids[-1]:
            vertex_ids.pop()
        return vertex_ids

    def _order_edges(self, edges: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        if not edges:
            raise ValueError("no edges provided")

        remaining = edges[:]
        ordered = [remaining.pop(0)]
        guard = 0
        while remaining and guard < 1000:
            guard += 1
            last_end = ordered[-1][1]
            for index, (start, end) in enumerate(remaining):
                if start == last_end:
                    ordered.append((start, end))
                    remaining.pop(index)
                    break
                if end == last_end:
                    ordered.append((end, start))
                    remaining.pop(index)
                    break
            else:
                raise ValueError("non-manifold or disconnected loop")
        if remaining:
            raise ValueError("loop could not be closed")
        if ordered[-1][1] != ordered[0][0]:
            raise ValueError("loop does not form a closed ring")
        return ordered

    def _triangulate_polygon(self, polygon: Sequence[Tuple[float, float, float]]) -> List[Tuple[int, int, int]]:
        if len(polygon) < 3:
            raise ValueError("polygon requires at least three points")
        origin, axis_u, axis_v, normal = self._compute_basis(polygon)
        points_2d = [
            (
                _dot(_vector_sub(point, origin), axis_u),
                _dot(_vector_sub(point, origin), axis_v),
            )
            for point in polygon
        ]
        orientation = 1 if self._polygon_area(points_2d) >= 0 else -1
        return self._ear_clip(points_2d, orientation)

    def _compute_basis(self, polygon: Sequence[Tuple[float, float, float]]):
        origin = polygon[0]
        axis_u: Optional[Tuple[float, float, float]] = None
        for point in polygon[1:]:
            vec = _vector_sub(point, origin)
            if _vector_length(vec) > self._EPS:
                axis_u = _normalise(vec)
                break
        if axis_u is None:
            raise ValueError("degenerate polygon: cannot determine tangent axis")

        normal: Optional[Tuple[float, float, float]] = None
        for point in polygon[2:]:
            vec = _vector_sub(point, origin)
            candidate = _cross(axis_u, vec)
            if _vector_length(candidate) > self._EPS:
                normal = _normalise(candidate)
                break
        if normal is None:
            raise ValueError("degenerate polygon: cannot determine normal")

        axis_v = _normalise(_cross(normal, axis_u))
        return origin, axis_u, axis_v, normal

    def _polygon_area(self, points: Sequence[Tuple[float, float]]) -> float:
        area = 0.0
        for idx in range(len(points)):
            x1, y1 = points[idx]
            x2, y2 = points[(idx + 1) % len(points)]
            area += x1 * y2 - x2 * y1
        return area / 2.0

    def _ear_clip(self, points: Sequence[Tuple[float, float]], orientation: int) -> List[Tuple[int, int, int]]:
        indices = list(range(len(points)))
        triangles: List[Tuple[int, int, int]] = []
        guard = 0
        while len(indices) > 3 and guard < 1000:
            guard += 1
            ear_found = False
            for i, current in enumerate(indices):
                prev_index = indices[i - 1]
                next_index = indices[(i + 1) % len(indices)]
                if not self._is_convex(points[prev_index], points[current], points[next_index], orientation):
                    continue
                triangle = (prev_index, current, next_index)
                if self._triangle_contains_vertex(points, triangle, indices):
                    continue
                triangles.append(triangle)
                del indices[i]
                ear_found = True
                break
            if not ear_found:
                raise ValueError("unable to triangulate polygon; polygon may be self-intersecting")
        if len(indices) == 3:
            triangles.append(tuple(indices))
        return triangles

    def _is_convex(
        self,
        prev_point: Tuple[float, float],
        current_point: Tuple[float, float],
        next_point: Tuple[float, float],
        orientation: int,
    ) -> bool:
        cross = (
            (current_point[0] - prev_point[0]) * (next_point[1] - current_point[1])
            - (current_point[1] - prev_point[1]) * (next_point[0] - current_point[0])
        )
        return cross * orientation > self._EPS

    def _triangle_contains_vertex(
        self,
        points: Sequence[Tuple[float, float]],
        triangle: Tuple[int, int, int],
        polygon_indices: Sequence[int],
    ) -> bool:
        a, b, c = triangle
        pa = points[a]
        pb = points[b]
        pc = points[c]
        for index in polygon_indices:
            if index in triangle:
                continue
            if _point_in_triangle(points[index], pa, pb, pc, self._EPS):
                return True
        return False


def _vector_sub(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vector_length(vec: Tuple[float, float, float]) -> float:
    return math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])


def _dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalise(vec: Tuple[float, float, float]) -> Tuple[float, float, float]:
    length = _vector_length(vec)
    if length == 0:
        raise ValueError("Cannot normalise zero-length vector")
    return (vec[0] / length, vec[1] / length, vec[2] / length)


def _point_in_triangle(
    point: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float],
    eps: float,
) -> bool:
    det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(det) < eps:
        return False
    l1 = ((b[1] - c[1]) * (point[0] - c[0]) + (c[0] - b[0]) * (point[1] - c[1])) / det
    l2 = ((c[1] - a[1]) * (point[0] - c[0]) + (a[0] - c[0]) * (point[1] - c[1])) / det
    l3 = 1.0 - l1 - l2
    return -eps <= l1 <= 1 + eps and -eps <= l2 <= 1 + eps and -eps <= l3 <= 1 + eps


def _triangle_normal(
    p0: Tuple[float, float, float],
    p1: Tuple[float, float, float],
    p2: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    u = _vector_sub(p1, p0)
    v = _vector_sub(p2, p0)
    normal = _cross(u, v)
    length = _vector_length(normal)
    if length == 0:
        return (0.0, 0.0, 0.0)
    return (normal[0] / length, normal[1] / length, normal[2] / length)


__docformat__ = "restructuredtext"
