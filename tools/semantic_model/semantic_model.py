from __future__ import annotations

import json
import re
import unicodedata
import uuid
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import Element, SubElement, tostring

from rdflib import Graph, Literal, Namespace, OWL, RDF, RDFS, SKOS, URIRef, XSD

from resources.semantic_model.utils import MODELS_PATH


BASE_MODELS_PATH = Path(MODELS_PATH)
BASE_MODELS_PATH.mkdir(parents=True, exist_ok=True)

UML_META = Namespace("urn:harmonia:uml:")


def ea_id(prefix: str, uri: str) -> str:
    u = uuid.uuid5(uuid.NAMESPACE_URL, str(uri))
    s = str(u).replace("-", "_").upper()
    return f"{prefix}_{s}"


def local_name(uri: URIRef | str) -> str:
    s = str(uri)
    if "#" in s:
        return s.split("#")[-1]
    return s.rstrip("/").split("/")[-1]


def _norm_text(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value if value else None


def _safe_xmi_value(value: Any) -> str:
    value = _norm_text(value)
    return value if value is not None else ""


def _canonical_text(value: Any) -> str:
    value = _norm_text(value) or ""
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"\s+", " ", value).strip().casefold()
    return value


def _display_name_from_stored(stored_name: str) -> str:
    """Return the human-readable part of a timestamped stored model name."""
    if not stored_name:
        return "Generated"
    if "__" in stored_name:
        return stored_name.rsplit("__", 1)[0]
    return stored_name


def _slugify_uri(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return value or "generated"


def _sanitize_path_component(value: str | None, default: str) -> str:
    value = _norm_text(value) or default
    value = value.replace("\\", "_").replace("/", "_").replace("\x00", "")
    return value.strip() or default


def _find_file(user: str, name: str) -> Path:
    safe_user = _sanitize_path_component(user, "default")
    safe_name = _sanitize_path_component(name, "generated")
    return BASE_MODELS_PATH / safe_user / f"{safe_name}.json"


def get_model_path(user: str = "", name: str = "") -> str:
    return str(_find_file(user, name))


def _model_uri(user: str, name: str) -> URIRef:
    u = _slugify_uri(user or 'default')
    n = _slugify_uri(name or 'generated')
    return URIRef(f"urn:harmonia:model:{u}:{n}")


def _new_graph(user: str, name: str) -> Graph:
    g = Graph()
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("skos", SKOS)
    g.bind("uml", UML_META)

    onto = _model_uri(user, name)
    g.add((onto, RDF.type, OWL.Ontology))
    g.add((onto, RDFS.label, Literal(name or "Generated", lang="fr")))
    return g


def _first_literal(
    g: Graph,
    s: URIRef,
    p: URIRef,
    preferred_langs: tuple[str, ...] = ("fr", "en"),
) -> str | None:
    vals = list(g.objects(s, p))
    if not vals:
        return None

    for lang in preferred_langs:
        for v in vals:
            if isinstance(v, Literal) and v.language == lang:
                return str(v)

    for v in vals:
        if isinstance(v, Literal):
            return str(v)

    return str(vals[0])


def get_label(g: Graph, s: URIRef) -> str | None:
    return _first_literal(g, s, RDFS.label)


def get_comment(g: Graph, s: URIRef) -> str | None:
    return _first_literal(g, s, RDFS.comment)


def get_scope_note(g: Graph, s: URIRef) -> str | None:
    return _first_literal(g, s, SKOS.scopeNote)


def get_literal_value(g: Graph, s: URIRef, p: URIRef) -> str | None:
    vals = list(g.objects(s, p))
    if not vals:
        return None
    return str(vals[0])


def _set_literal(
    g: Graph,
    s: URIRef,
    p: URIRef,
    value: str | None,
    lang: str | None = None,
) -> None:
    g.remove((s, p, None))
    value = _norm_text(value)
    if value is not None:
        if lang:
            g.add((s, p, Literal(value, lang=lang)))
        else:
            g.add((s, p, Literal(value)))


def _set_literal_if_absent(
    g: Graph,
    s: URIRef,
    p: URIRef,
    value: str | None,
    lang: str | None = None,
) -> None:
    if list(g.objects(s, p)):
        return
    value = _norm_text(value)
    if value is not None:
        if lang:
            g.add((s, p, Literal(value, lang=lang)))
        else:
            g.add((s, p, Literal(value)))


def _add_literal(
    g: Graph,
    s: URIRef,
    p: URIRef,
    value: str | None,
    lang: str | None = None,
) -> None:
    value = _norm_text(value)
    if value is not None:
        if lang:
            g.add((s, p, Literal(value, lang=lang)))
        else:
            g.add((s, p, Literal(value)))


def _set_uri(g: Graph, s: URIRef, p: URIRef, value: URIRef | str | None) -> None:
    g.remove((s, p, None))
    if value:
        g.add((s, p, URIRef(str(value))))


def _add_uri(g: Graph, s: URIRef, p: URIRef, value: URIRef | str | None) -> None:
    if value:
        g.add((s, p, URIRef(str(value))))


def _coerce_class_uri(value: str | None, fallback_label: str) -> URIRef:
    value = _norm_text(value)
    if value and (
                  value.startswith("http://") or
                  value.startswith("https://") or
                  value.startswith("urn:")
                  ):
        return URIRef(value)
    if value:
        return URIRef(f"urn:eaid:{value}")
    return URIRef(f"urn:class:{_slugify_uri(fallback_label)}")


def primitive_from_range(r: URIRef) -> str | None:
    s = str(r)
    if s in (str(RDFS.Literal), str(RDF.langString)):
        return "String"

    xsd_map = {
        str(XSD.string): "String",
        str(XSD.boolean): "Boolean",
        str(XSD.integer): "Integer",
        str(XSD.int): "Integer",
        str(XSD.long): "Integer",
        str(XSD.float): "Real",
        str(XSD.double): "Real",
        str(XSD.decimal): "Real",
        str(XSD.date): "Date",
        str(XSD.dateTime): "DateTime",
        str(XSD.time): "Time",
        str(XSD.anyURI): "URI",
    }
    return xsd_map.get(s)


_FRIENDLY_TO_XSD_URI: dict[str, str] = {
    "string": str(XSD.string),
    "text": str(RDF.langString),
    "literal": str(RDFS.Literal),
    "boolean": str(XSD.boolean),
    "integer": str(XSD.integer),
    "int": str(XSD.int),
    "long": str(XSD.long),
    "float": str(XSD.float),
    "real": str(XSD.double),
    "double": str(XSD.double),
    "decimal": str(XSD.decimal),
    "number": str(XSD.decimal),
    "date": str(XSD.date),
    "datetime": str(XSD.dateTime),
    "time": str(XSD.time),
    "uri": str(XSD.anyURI),
    "xsd:string": str(XSD.string),
    "xsd:boolean": str(XSD.boolean),
    "xsd:integer": str(XSD.integer),
    "xsd:decimal": str(XSD.decimal),
    "xsd:float": str(XSD.float),
    "xsd:double": str(XSD.double),
    "xsd:date": str(XSD.date),
    "xsd:dateTime": str(XSD.dateTime),
    "xsd:anyURI": str(XSD.anyURI),
    "rdf:langString": str(RDF.langString),
    "rdfs:Literal": str(RDFS.Literal),
}


def resolve_type_uri(attr_type: str) -> tuple[str, bool]:
    if not attr_type:
        return ("", False)

    if "://" in attr_type or attr_type.startswith("urn:"):
        prim = primitive_from_range(URIRef(attr_type))
        return (attr_type, prim is not None)

    canonical = _FRIENDLY_TO_XSD_URI.get(attr_type) or _FRIENDLY_TO_XSD_URI.get(attr_type.lower())
    if canonical:
        return (canonical, True)

    return (attr_type, False)


def _default_source_format(model: dict[str, Any]) -> str:
    fmt = _norm_text(model.get("source_format"))
    if fmt:
        return fmt.lower()

    if model.get("ttl_raw"):
        return "ttl"
    return "xmi"


def extract_ontology_package(g: Graph, custom_name: str | None = None) -> dict[str, Any]:
    ontos = list(g.subjects(RDF.type, OWL.Ontology))
    if ontos:
        onto = ontos[0]
        name = custom_name or get_label(g, onto) or local_name(onto)
        pkg_uri = str(onto)
    else:
        name = custom_name or "Generated"
        pkg_uri = f"urn:pkg:{name}"

    return {
        "name": name,
        "ID": ea_id("EAPK", pkg_uri),
        "type": "uml:Package",
        "package": "",
        "uri": pkg_uri,
        "tags": [],
    }


def _first_tag_value(tags: list[dict[str, Any]] | None, key: str) -> str:
    for tag in tags or []:
        if _norm_text(tag.get("name")) == key:
            return _norm_text(tag.get("value")) or ""
    return ""


def _xmi_view(model: dict[str, Any]) -> dict[str, Any]:
    xmi = model.get("xmi")
    if isinstance(xmi, dict):
        return {
            "elements": xmi.get("elements", []) or [],
            "connectors": xmi.get("connectors", []) or [],
        }
    return {
        "elements": model.get("elements", []) or [],
        "connectors": model.get("connectors", []) or [],
    }


def _class_uri_from_element(el: dict[str, Any]) -> URIRef:
    uri = _norm_text(el.get("uri"))
    if uri:
        return URIRef(uri)

    uri = _first_tag_value(el.get("tags", []), "uri")
    if uri:
        return URIRef(uri)

    el_id = _norm_text(el.get("ID"))
    if el_id:
        return URIRef(f"urn:eaid:{el_id}")

    return URIRef(
        f"urn:class:{_slugify_uri(_safe_xmi_value(el.get('name')) or 'generated')}"
    )


def _attribute_uri(class_uri: URIRef, attr: dict[str, Any]) -> URIRef:
    uri = _norm_text(attr.get("uri"))
    if uri:
        return URIRef(uri)

    uri = _first_tag_value(attr.get("tags_attribute", []), "uri")
    if uri:
        return URIRef(uri)

    attr_name = _safe_xmi_value(attr.get("name")) or "attribute"
    return URIRef(f"{str(class_uri)}#{_slugify_uri(attr_name)}")


def _semantic_uri_from_connector_view(conn: dict[str, Any]) -> str:
    return (
        _norm_text(conn.get("uri"))
        or _norm_text(conn.get("semantic_uri"))
        or _first_tag_value(conn.get("tags", []), "semantic_uri")
        or _first_tag_value(conn.get("tags_target", []), "semantic_uri")
        or _first_tag_value(conn.get("tags_source", []), "semantic_uri")
        or _first_tag_value(conn.get("tags", []), "uri")
        or _first_tag_value(conn.get("tags_target", []), "uri")
        or _first_tag_value(conn.get("tags_source", []), "uri")
        or ""
    )


def _connector_id_from_view(conn: dict[str, Any]) -> str:
    return (
        _norm_text(conn.get("connector_id"))
        or _first_tag_value(conn.get("tags", []), "connector_id")
        or _first_tag_value(conn.get("tags_target", []), "connector_id")
        or _first_tag_value(conn.get("tags_source", []), "connector_id")
        or ""
    )


def _connector_key_from_parts(
    semantic_uri: str,
    relationship: str,
    source_uri: str,
    target_uri: str,
) -> str:
    return "§".join(
        [
            _norm_text(semantic_uri) or "",
            _norm_text(relationship) or "",
            _norm_text(source_uri) or "",
            _norm_text(target_uri) or "",
        ]
    )


def _build_connector_id(
    semantic_uri: str,
    relationship: str,
    source_uri: str,
    target_uri: str,
) -> str:
    return ea_id(
        "CONN",
        _connector_key_from_parts(semantic_uri, relationship, source_uri, target_uri),
    )


def _resolve_class_uri_from_ref(
    *,
    ref_id: str = "",
    ref_name: str = "",
    class_uri_by_id: dict[str, URIRef],
    class_uri_by_name: dict[str, URIRef],
) -> URIRef | None:
    ref_id = _norm_text(ref_id) or ""
    ref_name = _norm_text(ref_name) or ""

    if ref_id and ref_id in class_uri_by_id:
        return class_uri_by_id[ref_id]

    if ref_name:
        found = class_uri_by_name.get(_canonical_text(ref_name))
        if found:
            return found

    if ref_id:
        return URIRef(f"urn:eaid:{ref_id}")

    if ref_name:
        return URIRef(f"urn:class:{_slugify_uri(ref_name)}")

    return None


def _extract_connector_overrides(model: dict[str, Any]) -> dict[str, dict[str, Any]]:
    xmi = _xmi_view(model)
    elements = xmi.get("elements", [])
    connectors = xmi.get("connectors", [])

    class_uri_by_id: dict[str, str] = {}
    class_uri_by_name: dict[str, str] = {}

    for el in elements:
        if _norm_text(el.get("type")) != "uml:Class":
            continue

        class_uri = str(_class_uri_from_element(el))
        el_id = _safe_xmi_value(el.get("ID"))
        el_name = _safe_xmi_value(el.get("name"))

        if el_id:
            class_uri_by_id[el_id] = class_uri
        if el_name:
            class_uri_by_name[_canonical_text(el_name)] = class_uri

    overrides: dict[str, dict[str, Any]] = {}

    for conn in connectors:
        relationship = _norm_text(conn.get("relationship")) or "Association"

        source_uri = (
            class_uri_by_id.get(_safe_xmi_value(conn.get("source_id")))
            or class_uri_by_name.get(_canonical_text(conn.get("source_name")))
            or ""
        )
        target_uri = (
            class_uri_by_id.get(_safe_xmi_value(conn.get("target_id")))
            or class_uri_by_name.get(_canonical_text(conn.get("target_name")))
            or ""
        )

        semantic_uri = _semantic_uri_from_connector_view(conn)
        if relationship == "Generalization":
            semantic_uri = ""

        if not source_uri or not target_uri:
            continue

        key = _connector_key_from_parts(semantic_uri, relationship, source_uri, target_uri)
        connector_id = _connector_id_from_view(conn) or _build_connector_id(
            semantic_uri,
            relationship,
            source_uri,
            target_uri,
        )

        overrides[key] = {
            "connector_id": connector_id,
            "semantic_uri": semantic_uri,
            "relationship": relationship,
            "name": _safe_xmi_value(conn.get("name")),
            "lb": _safe_xmi_value(conn.get("lb")),
            "lt": _safe_xmi_value(conn.get("lt")),
            "rb": _safe_xmi_value(conn.get("rb")),
            "rt": _safe_xmi_value(conn.get("rt")),
            "source_name": _safe_xmi_value(conn.get("source_name")),
            "target_name": _safe_xmi_value(conn.get("target_name")),
        }

    return overrides


def role_name_from_label(label: str | None) -> str | None:
    if not label:
        return None
    return f"+{label}"


def build_model(
    g: Graph,
    package_name: str | None = None,
    connector_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    model: dict[str, Any] = {"elements": [], "connectors": []}
    connector_overrides = connector_overrides or {}

    root_pkg = extract_ontology_package(g, custom_name=package_name)
    model["elements"].append(root_pkg)
    root_pkg_id = root_pkg["ID"]

    class_elems: dict[str, dict[str, Any]] = {}

    for cls in g.subjects(RDF.type, OWL.Class):
        uri = str(cls)
        name = get_label(g, cls) or local_name(cls)
        stored_id = get_literal_value(g, cls, UML_META.eaid) or ea_id("EAID", uri)

        tags = [
            {"name": "uri", "value": uri},
            {"name": "referenced", "value": "false"},
        ]
        definition = get_comment(g, cls)
        usage_note = get_scope_note(g, cls)
        label = get_label(g, cls)

        if definition:
            tags.append({"name": "definition-fr", "value": definition})
        if label:
            tags.append({"name": "label-fr", "value": label})
        if usage_note:
            tags.append({"name": "usageNote-fr", "value": usage_note})

        elem = {
            "name": name,
            "ID": stored_id,
            "type": "uml:Class",
            "package": root_pkg_id,
            "uri": uri,
            "tags": tags,
            "attributes": [],
        }
        class_elems[uri] = elem
        model["elements"].append(elem)

    def ensure_class(uri_ref: URIRef) -> dict[str, Any]:
        u = str(uri_ref)
        if primitive_from_range(uri_ref) is not None:
            raise ValueError(f"Primitive URI passed to ensure_class: {u}")

        if u not in class_elems:
            name = get_label(g, uri_ref) or local_name(uri_ref)
            stored_id = get_literal_value(g, uri_ref, UML_META.eaid) or ea_id("EAID", u)
            elem = {
                "name": name,
                "ID": stored_id,
                "type": "uml:Class",
                "package": root_pkg_id,
                "uri": u,
                "tags": [
                    {"name": "uri", "value": u},
                    {"name": "referenced", "value": "true"},
                ],
                "attributes": [],
            }
            class_elems[u] = elem
            model["elements"].append(elem)

        return class_elems[u]

    # Séparation intelligente des attributs et connecteurs
    # Pour garder les attributs de type "complexe" dans la boîte de classe visuellement.
    attribute_props = set(g.subjects(RDF.type, OWL.DatatypeProperty))
    connector_props = set()

    for prop in g.subjects(RDF.type, OWL.ObjectProperty):
        # S'il a un relationshipType explicite, c'est un connecteur (ligne)
        rel_type = get_literal_value(g, prop, UML_META.relationshipType)
        if rel_type:
            connector_props.add(prop)
        else:
            # Sinon, c'est un attribut complexe (ex: type "Entreprise") qui reste dans la classe
            attribute_props.add(prop)

    for prop in attribute_props:
        prop_uri = str(prop)
        prop_label = get_label(g, prop) or local_name(prop)
        prop_comment = get_comment(g, prop)
        prop_usage = get_scope_note(g, prop)

        for domain in g.objects(prop, RDFS.domain):
            domain_elem = ensure_class(domain)
            ranges = list(g.objects(prop, RDFS.range)) or [RDFS.Literal]

            for rng in ranges:
                primitive = primitive_from_range(rng)
                attr_type = primitive if primitive else ensure_class(rng)["name"]

                attr_tags = [{"name": "uri", "value": prop_uri}]
                if prop_label:
                    attr_tags.append({"name": "label-fr", "value": prop_label})
                if prop_comment:
                    attr_tags.append({"name": "definition-fr", "value": prop_comment})
                if prop_usage:
                    attr_tags.append({"name": "usageNote-fr", "value": prop_usage})

                lower_bounds = _safe_xmi_value(get_literal_value(g, prop, UML_META.lowerBound))
                upper_bounds = _safe_xmi_value(get_literal_value(g, prop, UML_META.upperBound))

                domain_elem["attributes"].append(
                    {
                        "name": prop_label,
                        "ID": ea_id("ATTR", prop_uri),
                        "type": attr_type,
                        "uri": prop_uri,
                        "lower_bounds": lower_bounds,
                        "upper_bounds": upper_bounds,
                        "tags_attribute": attr_tags,
                    }
                )

    for prop in connector_props:
        prop_uri = str(prop)
        prop_label = get_label(g, prop) or local_name(prop)
        prop_comment = get_comment(g, prop)
        prop_usage = get_scope_note(g, prop)

        domains = list(g.objects(prop, RDFS.domain))
        ranges = list(g.objects(prop, RDFS.range))
        if not domains or not ranges:
            continue

        relationship = get_literal_value(g, prop, UML_META.relationshipType) or "Association"
        default_lb = _safe_xmi_value(get_literal_value(g, prop, UML_META.leftMultiplicity))
        default_lt = _safe_xmi_value(get_literal_value(g, prop, UML_META.leftRole))
        default_rb = _safe_xmi_value(get_literal_value(g, prop, UML_META.rightMultiplicity))
        default_rt = _safe_xmi_value(get_literal_value(g, prop, UML_META.rightRole))

        for domain in domains:
            src = ensure_class(domain)
            for rng in ranges:
                tgt = ensure_class(rng)
                key = _connector_key_from_parts(prop_uri, relationship, str(domain), str(rng))
                override = connector_overrides.get(key, {})
                connector_id = override.get("connector_id") or _build_connector_id(
                    prop_uri,
                    relationship,
                    str(domain),
                    str(rng),
                )

                tgt_tags = [
                    {"name": "uri", "value": prop_uri},
                    {"name": "semantic_uri", "value": prop_uri},
                    {"name": "connector_id", "value": connector_id},
                ]
                if prop_comment:
                    tgt_tags.append({"name": "definition-fr", "value": prop_comment})
                if prop_label:
                    tgt_tags.append({"name": "label-fr", "value": prop_label})
                if prop_usage:
                    tgt_tags.append({"name": "usageNote-fr", "value": prop_usage})

                # Fusion intelligente : priorise l'override, sinon utilise la donnée RDF
                lb_val = override.get("lb")
                lb_val = lb_val if lb_val else default_lb

                lt_val = override.get("lt")
                lt_val = lt_val if lt_val else default_lt

                rb_val = override.get("rb")
                rb_val = rb_val if rb_val else default_rb

                rt_val = override.get("rt")
                rt_val = rt_val if rt_val else default_rt

                model["connectors"].append(
                    {
                        "connector_id": connector_id,
                        "semantic_uri": prop_uri,
                        "uri": prop_uri,
                        "source_name": override.get("source_name") or src["name"],
                        "source_id": src["ID"],
                        "target_name": override.get("target_name") or tgt["name"],
                        "target_id": tgt["ID"],
                        "relationship": relationship,
                        "name": override.get("name") or prop_label,
                        "lb": lb_val,
                        "lt": lt_val,
                        "rb": rb_val,
                        "rt": rt_val,
                        "tags": [{"name": "connector_id", "value": connector_id}],
                        "tags_source": [],
                        "tags_target": tgt_tags,
                    }
                )

    for child, parent in g.subject_objects(RDFS.subClassOf):
        if not isinstance(parent, URIRef):
            continue

        child_elem = ensure_class(child)
        parent_elem = ensure_class(parent)
        relationship = "Generalization"
        key = _connector_key_from_parts("", relationship, str(child), str(parent))
        override = connector_overrides.get(key, {})
        connector_id = override.get("connector_id") or _build_connector_id(
            "",
            relationship,
            str(child),
            str(parent),
        )

        model["connectors"].append(
            {
                "connector_id": connector_id,
                "semantic_uri": "",
                "uri": "",
                "source_name": override.get("source_name") or child_elem["name"],
                "source_id": child_elem["ID"],
                "target_name": override.get("target_name") or parent_elem["name"],
                "target_id": parent_elem["ID"],
                "relationship": relationship,
                "name": override.get("name") or "subClassOf",
                "lb": override.get("lb", ""),
                "lt": override.get("lt", ""),
                "rb": override.get("rb", ""),
                "rt": override.get("rt", ""),
                "tags": [{"name": "connector_id", "value": connector_id}],
                "tags_source": [],
                "tags_target": [{"name": "connector_id", "value": connector_id}],
            }
        )

    return model


def _sync_model_from_graph(
    g: Graph,
    model: dict[str, Any],
    package_name: str | None = None,
    source_format: str | None = None,
    connector_overrides: dict[str, dict[str, Any]] | None = None,
    keep_raw: bool = False,
    update_elements: bool = True,
) -> dict[str, Any]:
    print(f"[DEBUG _sync_model_from_graph] Début reconstruction du modèle depuis le graphe RDF ({len(g)} triplets)...")

    json_ld_str = g.serialize(format="json-ld", indent=4)
    ttl_raw = g.serialize(format="turtle")
    
    if isinstance(json_ld_str, (bytes, bytearray)):
        json_ld_str = json_ld_str.decode("utf-8")

    model["ttl"] = json.loads(json_ld_str) if json_ld_str else {}
    model["ttl_raw"] = ttl_raw.decode("utf-8") if isinstance(ttl_raw, (bytes, bytearray)) else str(ttl_raw)

    if update_elements:
        xmi = build_model(
            g,
            package_name=package_name,
            connector_overrides=connector_overrides or _extract_connector_overrides(model),
        )
        model["xmi"] = xmi
        model["elements"] = xmi.get("elements", [])
        model["connectors"] = xmi.get("connectors", [])

    model["source_format"] = (source_format or model.get("source_format") or "ttl").lower()

    if not keep_raw:
        for key in ["xmi_raw", "xmi_xml", "xmiraw", "xmixml"]:
            model.pop(key, None)

    print(f"[DEBUG _sync_model_from_graph] Modèle mis à jour avec {len(model.get('elements', []))} éléments et {len(model.get('connectors', []))} connecteurs.")
    return model


def _save_model(fp: Path, model: dict[str, Any], user: str = "", name: str = "") -> dict[str, Any]:
    """
    Sauvegarde uniquement le modèle JSON principal en forçant l'UTF-8 de manière stricte (ensure_ascii=False).
    Les fichiers .ttl et .xmi ne polluent plus le disque, ils sont générés dynamiquement par l'exportation.
    """
    print(f"[DEBUG _save_model] Début de la sauvegarde JSON dans : {fp}")
    fp.parent.mkdir(parents=True, exist_ok=True)
    
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)
    print(f"[DEBUG _save_model] Fichier JSON sauvegardé proprement en UTF-8.")

    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
        print(f"[DEBUG _save_model] Re-lecture de contrôle réussie. Éléments relus : {len(data.get('elements', []))}")
        return data


def _load_model(fp: Path) -> dict[str, Any]:
    print(f"[DEBUG _load_model] Tentative de chargement du fichier : {fp}")
    if not fp.exists():
        print(f"[DEBUG _load_model] Le fichier n'existe pas encore.")
        return {}

    try:
        with open(fp, "r", encoding="utf-8") as f:
            raw = f.read().strip()
            if not raw:
                print(f"[DEBUG _load_model] Fichier vide détecté.")
                return {}
            print(f"[DEBUG _load_model] Succès de la lecture physique ({len(raw)} caractères).")
            data = json.loads(raw)
            return data
    except Exception as e:
        print(f"[DEBUG _load_model] ERREUR critique lors du chargement : {e}")
        return {}


def _ensure_synchronized(model: dict[str, Any], user: str, name: str, keep_raw: bool = True) -> dict[str, Any]:
    """
    Synchronise le modèle sémantique.
    Mise à jour essentielle : si `has_elements` est True (via EA parser), on n'écrase plus les éléments
    graphiques/diagrammes pour ne générer QUE le ttl_raw d'exportation (update_elements=False).

    `name` is the technical stored file name (may contain a timestamp suffix).
    We always derive the human-readable package/ontology label from the JSON
    `name` field or by stripping that suffix, so exports never leak technical
    identifiers into the generated RDF.
    """
    print(f"[DEBUG _ensure_synchronized] Début de la vérification de synchronisation...")
    if not model:
        return model

    raw_xmi = ""
    for key in ("xmi_raw", "xmi_xml", "xmiraw", "xmixml"):
        candidate = model.get(key) or ""
        if isinstance(candidate, str) and candidate.lstrip().startswith("<?xml"):
            raw_xmi = candidate
            break

    has_raw_xmi = bool(raw_xmi)
    has_ttl = bool(_norm_text(model.get("ttl_raw")))
    has_elements = bool(_xmi_view(model).get("elements"))

    print(f"[DEBUG _ensure_synchronized] Status : has_raw_xmi={has_raw_xmi}, has_ttl={has_ttl}, has_elements={has_elements}")

    needs_sync = (
        (has_raw_xmi and (not has_ttl or not has_elements)) or
        (has_elements and not has_ttl) or
        (has_ttl and not has_elements)
    )

    if needs_sync:
        print(f"[DEBUG _ensure_synchronized] Synchronisation nécessaire.")
        # Never use the technical file name as the ontology/package label.
        display_name = _norm_text(model.get("name")) or _display_name_from_stored(name)
        g = graph_from_model(model, user=user, name=display_name)
        # On évite d'écraser les diagrammes/éléments complexes provenant de l'import
        update_elements = not has_elements
        model = _sync_model_from_graph(
            g,
            model,
            package_name=display_name or "Generated",
            source_format="xmi" if has_raw_xmi else "ttl",
            keep_raw=keep_raw,
            update_elements=update_elements
        )
        # Ensure the model remembers its display name for future exports.
        model["name"] = display_name
        fp = _find_file(user, name)
        model = _save_model(fp, model, user=user, name=name)
        print(f"[DEBUG _ensure_synchronized] Synchronisation terminée.")

    return model


def ensure_model_exists(user: str = "", name: str = "") -> Path:
    fp = _find_file(user, name)
    if fp.exists():
        return fp

    display_name = _display_name_from_stored(name)
    g = _new_graph(user, display_name)
    model: dict[str, Any] = {
        "name": display_name,
        "elements": [],
        "connectors": [],
        "xmi": {"elements": [], "connectors": []},
        "ttl": {},
        "ttl_raw": "",
        "source_format": "ttl",
    }
    model = _sync_model_from_graph(g, model, package_name=display_name, source_format="ttl", keep_raw=False, update_elements=True)
    _save_model(fp, model, user=user, name=name)
    return fp


def build_graph_from_xmi_model(
    model: dict[str, Any],
    user: str = "",
    name: str = "",
) -> Graph:
    import xml.etree.ElementTree as ET

    NS_XMI = "http://schema.omg.org/spec/XMI/2.1"
    NS_UML = "http://schema.omg.org/spec/UML/2.1"

    def _local(tag: str) -> str:
        return tag.split("}", 1)[1] if tag.startswith("{") else tag

    def _attr(el: ET.Element, name: str, default: str = "") -> str:
        return _norm_text(el.get(name)) or default

    def _xmi_attr(el: ET.Element, name: str, default: str = "") -> str:
        return _norm_text(el.get(f"{{{NS_XMI}}}{name}")) or default

    def _find_direct_children(el: ET.Element, local_name: str) -> list[ET.Element]:
        return [child for child in list(el) if _local(child.tag) == local_name]

    def _find_first_direct_child(el: ET.Element, local_name: str) -> ET.Element | None:
        for child in list(el):
            if _local(child.tag) == local_name:
                return child
        return None

    def _comment_body(el: ET.Element) -> str:
        parts: list[str] = []
        for c in _find_direct_children(el, "ownedComment"):
            body = _find_first_direct_child(c, "body")
            if body is not None and _norm_text(body.text):
                parts.append(_norm_text(body.text) or "")
        return "\n".join(p for p in parts if p).strip()

    def _parse_semantic_comment(body: str) -> dict[str, str]:
        out: dict[str, str] = {}

        mapping = {
            "uri": "uri",
            "label": "label",
            "definition": "definition",
            "usage note": "usage_note",
            "usage_note": "usage_note",
            "usagenote": "usage_note",
            "referenced": "referenced",
            "connector id": "connector_id",
            "connector_id": "connector_id",
        }

        for raw_line in (body or "").splitlines():
            line = (raw_line or "").strip()
            if not line or ":" not in line:
                continue

            key, value = line.split(":", 1)
            k = _canonical_text(key)
            field = mapping.get(k)
            if not field:
                continue

            v = _norm_text(value)
            if not v:
                continue

            md_link = re.match(r"^\[([^\]]+)\]\(([^)]+)\)$", v)
            if md_link:
                if field == "uri":
                    v = md_link.group(2).strip()
                elif field == "label":
                    v = md_link.group(1).strip()

            out[field] = v

        return out

    def _classifier_uri_from_xml(el: ET.Element) -> URIRef:
        body = _comment_body(el)
        meta = _parse_semantic_comment(body)
        uri = _norm_text(meta.get("uri"))
        if uri:
            return URIRef(uri)

        el_id = _xmi_attr(el, "id")
        el_name = _attr(el, "name")
        if el_id:
            return URIRef(f"urn:eaid:{el_id}")
        return URIRef(f"urn:class:{_slugify_uri(el_name or 'generated')}")

    def _property_uri_from_xml(owner_uri: URIRef, prop_el: ET.Element) -> URIRef:
        body = _comment_body(prop_el)
        meta = _parse_semantic_comment(body)
        uri = _norm_text(meta.get("uri"))
        if uri:
            return URIRef(uri)

        prop_name = _attr(prop_el, "name") or "attribute"
        return URIRef(f"{str(owner_uri)}#{_slugify_uri(prop_name)}")

    def _resolve_type_from_owned_attribute(
        prop_el: ET.Element,
        classifier_uri_by_id: dict[str, URIRef],
    ) -> tuple[str | None, bool]:
        primitive_map = {
            "string": str(XSD.string),
            "boolean": str(XSD.boolean),
            "integer": str(XSD.integer),
            "int": str(XSD.int),
            "long": str(XSD.long),
            "real": str(XSD.double),
            "double": str(XSD.double),
            "float": str(XSD.float),
            "decimal": str(XSD.decimal),
            "number": str(XSD.decimal),
            "date": str(XSD.date),
            "datetime": str(XSD.dateTime),
            "time": str(XSD.time),
            "uri": str(XSD.anyURI),
            "anyuri": str(XSD.anyURI),
        }

        def _normalize_type_name(value: str | None) -> str | None:
            raw = _norm_text(value)
            if not raw:
                return None

            token = raw.split("#")[-1].split("/")[-1].split(":")[-1].strip()
            key = token.lower()
            return primitive_map.get(key)

        def _resolve_type_token(value: str | None) -> tuple[str | None, bool] | None:
            raw = _norm_text(value)
            if not raw:
                return None

            if raw == "uml:Property":
                return None

            if raw in classifier_uri_by_id:
                return (str(classifier_uri_by_id[raw]), False)

            primitive_uri = _normalize_type_name(raw)
            if primitive_uri:
                return (primitive_uri, True)

            if raw.startswith("http://") or raw.startswith("https://") or raw.startswith("urn:"):
                if raw in {
                    str(XSD.string),
                    str(XSD.boolean),
                    str(XSD.integer),
                    str(XSD.int),
                    str(XSD.long),
                    str(XSD.float),
                    str(XSD.double),
                    str(XSD.decimal),
                    str(XSD.date),
                    str(XSD.dateTime),
                    str(XSD.time),
                    str(XSD.anyURI),
                }:
                    return (raw, True)
                return (raw, False)

            return None

        type_child = _find_first_direct_child(prop_el, "type")
        if type_child is not None:
            for k, v in type_child.attrib.items():
                local_k = k.split("}", 1)[1] if k.startswith("{") else k
                if local_k in {"idref", "href", "id", "type", "name"}:
                    resolved = _resolve_type_token(v)
                    if resolved is not None:
                        return resolved

        for candidate in (
            _attr(prop_el, "datatype"),
            _attr(prop_el, "classifier"),
            _attr(prop_el, "type"),
        ):
            resolved = _resolve_type_token(candidate)
            if resolved is not None:
                return resolved

        for candidate in (
            _attr(prop_el, "name"),
            _attr(type_child, "name") if type_child is not None else None,
        ):
            primitive_uri = _normalize_type_name(candidate)
            if primitive_uri:
                return (primitive_uri, True)

        return (str(XSD.string), True)

    def _resolve_end_type_id(end_el: ET.Element) -> str:
        tid = _attr(end_el, "type")
        if tid:
            return tid

        type_child = _find_first_direct_child(end_el, "type")
        if type_child is not None:
            for k, v in type_child.attrib.items():
                local_k = k.split("}", 1)[1] if k.startswith("{") else k
                if local_k in {"idref", "href", "id"}:
                    val = _norm_text(v)
                    if val:
                        if "#" in val:
                            return val.split("#")[-1]
                        return val
        return ""

    def _parse_literal_value(el: ET.Element) -> str | None:
        value = _attr(el, "value")
        return value if value != "" else None

    def _parse_multiplicity_bounds(prop_el: ET.Element) -> tuple[str | None, str | None]:
        lower = None
        upper = None

        lower_el = _find_first_direct_child(prop_el, "lowerValue")
        upper_el = _find_first_direct_child(prop_el, "upperValue")

        if lower_el is not None:
            lower = _parse_literal_value(lower_el)
        if upper_el is not None:
            upper = _parse_literal_value(upper_el)

        # Fallback pour récupérer directement depuis les attributs XML 'lower' et 'upper'
        # Très courant lors de l'export d'outils UML comme Enterprise Architect
        if lower is None:
            l_val = _attr(prop_el, "lower")
            if l_val:
                lower = l_val
                
        if upper is None:
            u_val = _attr(prop_el, "upper")
            if u_val:
                upper = u_val

        return lower, upper

    def _compose_multiplicity(lower: str | None, upper: str | None) -> str:
        lower = _norm_text(lower)
        upper = _norm_text(upper)

        if not lower and not upper:
            return ""
        if lower and not upper:
            return lower
        if upper and not lower:
            return upper
        if lower == upper:
            return lower or ""
        return f"{lower}..{upper}"

    def _new_or_default_graph(package_name: str) -> Graph:
        g = _new_graph(user, package_name)
        onto = next(g.subjects(RDF.type, OWL.Ontology), None)
        if isinstance(onto, URIRef):
            _set_literal(g, onto, RDFS.label, package_name, lang="fr")
        return g

    raw_xmi = ""
    for key in ("xmi_raw", "xmi_xml", "xmiraw", "xmixml"):
        candidate = model.get(key) or ""
        if isinstance(candidate, str) and candidate.lstrip().startswith("<?xml"):
            raw_xmi = candidate
            break

    raw_xml_bytes = b""
    if raw_xmi:
        try:
            if isinstance(raw_xmi, str):
                cleaned_xmi = re.sub(
                    r'(<\?xml[^>]*?)encoding=["\'].*?["\']',
                    r'\1encoding="utf-8"',
                    raw_xmi,
                    count=1
                )
                raw_xml_bytes = cleaned_xmi.encode("utf-8")
            else:
                raw_xml_bytes = raw_xmi
            
            root = ET.fromstring(raw_xml_bytes)
            print(f"[DEBUG build_graph_from_xmi_model] Succès du parsing XML de l'import brut.")
        except Exception as e:
            print(f"[DEBUG build_graph_from_xmi_model] ERREUR critique lors du parsing XML : {e}")
            return _new_graph(user, name or _norm_text(model.get("name")) or "Generated")
    else:
        xmi = _xmi_view(model)
        elements = xmi.get("elements", [])
        connectors = xmi.get("connectors", [])

        package_name = (
            _norm_text(name)
            or _norm_text(model.get("name"))
            or next(
                (
                    _norm_text(el.get("name"))
                    for el in elements
                    if _norm_text(el.get("type")) == "uml:Package" and _norm_text(el.get("name"))
                ),
                None,
            )
            or "Generated"
        )

        g = _new_graph(user, package_name)
        onto = next(g.subjects(RDF.type, OWL.Ontology), None)
        if isinstance(onto, URIRef):
            _set_literal(g, onto, RDFS.label, package_name, lang="fr")

        class_uri_by_id: dict[str, URIRef] = {}
        class_uri_by_name: dict[str, URIRef] = {}

        def ensure_class(uri: URIRef, label: str = "", eaid_value: str = "") -> URIRef:
            g.add((uri, RDF.type, OWL.Class))
            if label:
                _set_literal(g, uri, RDFS.label, label, lang="fr")
            if eaid_value:
                _set_literal(g, uri, UML_META.eaid, eaid_value)
            return uri

        for el in elements:
            el_type = _norm_text(el.get("type")) or ""
            if el_type not in {"uml:Class", "uml:DataType", "uml:Enumeration"}:
                continue

            class_uri = _class_uri_from_element(el)
            class_name = _safe_xmi_value(el.get("name"))
            class_id = _safe_xmi_value(el.get("ID"))

            label = _first_tag_value(el.get("tags", []), "label-fr") or class_name
            definition = _first_tag_value(el.get("tags", []), "definition-fr")
            usage_note = _first_tag_value(el.get("tags", []), "usageNote-fr")

            ensure_class(class_uri, label=label or class_name, eaid_value=class_id)
            _set_literal(g, class_uri, RDFS.comment, definition, lang="fr")
            _set_literal(g, class_uri, SKOS.scopeNote, usage_note, lang="fr")

            if class_id:
                class_uri_by_id[class_id] = class_uri
            if class_name:
                class_uri_by_name[_canonical_text(class_name)] = class_uri
            if label:
                class_uri_by_name[_canonical_text(label)] = class_uri

        for el in elements:
            el_type = _norm_text(el.get("type")) or ""
            if el_type not in {"uml:Class", "uml:DataType", "uml:Enumeration"}:
                continue

            owner_uri = _class_uri_from_element(el)

            for attr in el.get("attributes", []) or []:
                prop_uri = _attribute_uri(owner_uri, attr)
                attr_name = _safe_xmi_value(attr.get("name"))
                attr_label = _first_tag_value(attr.get("tags_attribute", []), "label-fr") or attr_name
                attr_definition = _first_tag_value(attr.get("tags_attribute", []), "definition-fr")
                attr_usage = _first_tag_value(attr.get("tags_attribute", []), "usageNote-fr")
                attr_type = _norm_text(attr.get("type")) or ""

                if attr_type == "uml:Property":
                    attr_type = (
                        _norm_text(attr.get("datatype"))
                        or _norm_text(attr.get("range"))
                        or _norm_text(attr.get("classifier"))
                        or ""
                    )

                canonical_range_uri = None
                is_primitive = True

                if attr_type:
                    canonical_range_uri, is_primitive = resolve_type_uri(attr_type)

                g.remove((prop_uri, RDF.type, OWL.DatatypeProperty))
                g.remove((prop_uri, RDF.type, OWL.ObjectProperty))
                g.add(
                    (prop_uri, RDF.type, OWL.DatatypeProperty if is_primitive else OWL.ObjectProperty)
                )

                _set_literal(g, prop_uri, RDFS.label, attr_label, lang="fr")
                _set_literal(g, prop_uri, RDFS.comment, attr_definition, lang="fr")
                _set_literal(g, prop_uri, SKOS.scopeNote, attr_usage, lang="fr")
                _add_uri(g, prop_uri, RDFS.domain, owner_uri)

                _set_literal_if_absent(g, prop_uri, UML_META.lowerBound, _safe_xmi_value(attr.get("lower_bounds")))
                _set_literal_if_absent(g, prop_uri, UML_META.upperBound, _safe_xmi_value(attr.get("upper_bounds")))

                if canonical_range_uri:
                    if is_primitive:
                        _set_uri(g, prop_uri, RDFS.range, canonical_range_uri)
                    else:
                        target_uri = (
                            class_uri_by_id.get(canonical_range_uri)
                            or class_uri_by_name.get(_canonical_text(canonical_range_uri))
                        )

                        if target_uri is None:
                            if "://" in canonical_range_uri or canonical_range_uri.startswith("urn:"):
                                target_uri = URIRef(canonical_range_uri)
                            else:
                                target_uri = URIRef(f"urn:class:{_slugify_uri(canonical_range_uri)}")
                            ensure_class(target_uri, label=attr_type or canonical_range_uri)

                        _add_uri(g, prop_uri, RDFS.range, target_uri)

        for conn in connectors:
            relationship = _norm_text(conn.get("relationship")) or "Association"

            source_uri = _resolve_class_uri_from_ref(
                ref_id=_safe_xmi_value(conn.get("source_id")),
                ref_name=_safe_xmi_value(conn.get("source_name")),
                class_uri_by_id=class_uri_by_id,
                class_uri_by_name=class_uri_by_name,
            )
            target_uri = _resolve_class_uri_from_ref(
                ref_id=_safe_xmi_value(conn.get("target_id")),
                ref_name=_safe_xmi_value(conn.get("target_name")),
                class_uri_by_id=class_uri_by_id,
                class_uri_by_name=class_uri_by_name,
            )

            if source_uri is None or target_uri is None:
                continue

            ensure_class(
                source_uri,
                label=_safe_xmi_value(conn.get("source_name")) or local_name(source_uri),
                eaid_value=_safe_xmi_value(conn.get("source_id")),
            )
            ensure_class(
                target_uri,
                label=_safe_xmi_value(conn.get("target_name")) or local_name(target_uri),
                eaid_value=_safe_xmi_value(conn.get("target_id")),
            )

            if relationship == "Generalization":
                g.add((source_uri, RDFS.subClassOf, target_uri))
                continue

            semantic_uri = _semantic_uri_from_connector_view(conn)
            if not semantic_uri:
                semantic_uri = f"urn:relation:{_slugify_uri(_safe_xmi_value(conn.get('name')) or 'relation')}"

            prop_uri = URIRef(semantic_uri)
            rel_label = (
                _norm_text(conn.get("name"))
                or _first_tag_value(conn.get("tags_target", []), "label-fr")
                or _first_tag_value(conn.get("tags", []), "label-fr")
                or local_name(prop_uri)
            )
            rel_definition = (
                _first_tag_value(conn.get("tags_target", []), "definition-fr")
                or _first_tag_value(conn.get("tags", []), "definition-fr")
            )
            rel_usage = (
                _first_tag_value(conn.get("tags_target", []), "usageNote-fr")
                or _first_tag_value(conn.get("tags", []), "usageNote-fr")
            )

            g.remove((prop_uri, RDF.type, OWL.DatatypeProperty))
            g.add((prop_uri, RDF.type, OWL.ObjectProperty))

            _set_literal(g, prop_uri, RDFS.label, rel_label, lang="fr")
            _set_literal(g, prop_uri, RDFS.comment, rel_definition, lang="fr")
            _set_literal(g, prop_uri, SKOS.scopeNote, rel_usage, lang="fr")
            _add_uri(g, prop_uri, RDFS.domain, source_uri)
            _add_uri(g, prop_uri, RDFS.range, target_uri)

            _set_literal(g, prop_uri, UML_META.relationshipType, relationship)

            _set_literal_if_absent(g, prop_uri, UML_META.leftMultiplicity, _safe_xmi_value(conn.get("lb")))
            _set_literal_if_absent(g, prop_uri, UML_META.rightMultiplicity, _safe_xmi_value(conn.get("rb")))
            _set_literal_if_absent(g, prop_uri, UML_META.leftRole, _safe_xmi_value(conn.get("lt")))
            _set_literal_if_absent(g, prop_uri, UML_META.rightRole, _safe_xmi_value(conn.get("rt")))

        return g

    try:
        root = ET.fromstring(raw_xmi.encode("utf-8") if isinstance(raw_xmi, str) else raw_xmi)
    except Exception as e:
        print(f"[DEBUG build_graph_from_xmi_model] ERREUR critique lors du parsing XML : {e}")
        return _new_graph(user, name or _norm_text(model.get("name")) or "Generated")

    uml_model = None
    for child in root.iter():
        if _local(child.tag) == "Model":
            uml_model = child
            break

    package_name = (
        _norm_text(name)
        or _norm_text(model.get("name"))
        or (uml_model is not None and _attr(uml_model, "name"))
        or "Generated"
    )

    g = _new_or_default_graph(package_name)

    classifier_uri_by_id: dict[str, URIRef] = {}
    classifier_name_by_id: dict[str, str] = {}
    association_end_by_id: dict[str, ET.Element] = {}
    pending_attributes: list[tuple[URIRef, ET.Element]] = []
    pending_associations: list[ET.Element] = []
    pending_generalizations: list[tuple[URIRef, ET.Element]] = []

    for el in root.iter():
        if _local(el.tag) != "packagedElement":
            continue

        xmi_type = _xmi_attr(el, "type")
        xmi_id = _xmi_attr(el, "id")
        xmi_name = _attr(el, "name")

        if xmi_type not in {"uml:Class", "uml:DataType", "uml:Enumeration", "uml:Association", "uml:Package"}:
            continue

        if xmi_type == "uml:Package":
            continue

        if xmi_type in {"uml:Class", "uml:DataType", "uml:Enumeration"}:
            class_uri = _classifier_uri_from_xml(el)
            meta = _parse_semantic_comment(_comment_body(el))

            label = meta.get("label") or xmi_name or local_name(class_uri)
            definition = meta.get("definition")
            usage_note = meta.get("usage_note")

            g.add((class_uri, RDF.type, OWL.Class))
            _set_literal(g, class_uri, RDFS.label, label, lang="fr")
            _set_literal(g, class_uri, RDFS.comment, definition, lang="fr")
            _set_literal(g, class_uri, SKOS.scopeNote, usage_note, lang="fr")
            if xmi_id:
                _set_literal(g, class_uri, UML_META.eaid, xmi_id)
                classifier_uri_by_id[xmi_id] = class_uri
                classifier_name_by_id[xmi_id] = label

            for child in _find_direct_children(el, "ownedAttribute"):
                pending_attributes.append((class_uri, child))

            for child in _find_direct_children(el, "generalization"):
                pending_generalizations.append((class_uri, child))

            if xmi_type == "uml:Enumeration":
                for lit in _find_direct_children(el, "ownedLiteral"):
                    lit_name = _attr(lit, "name")
                    if not lit_name:
                        continue
                    enum_prop_uri = URIRef(f"{str(class_uri)}#{_slugify_uri(lit_name)}")
                    g.add((enum_prop_uri, RDF.type, OWL.DatatypeProperty))
                    _set_literal(g, enum_prop_uri, RDFS.label, lit_name, lang="fr")
                    _add_uri(g, enum_prop_uri, RDFS.domain, class_uri)
                    _set_uri(g, enum_prop_uri, RDFS.range, XSD.string)

        elif xmi_type == "uml:Association":
            pending_associations.append(el)
            for end_el in _find_direct_children(el, "ownedEnd"):
                end_id = _xmi_attr(end_el, "id")
                if end_id:
                    association_end_by_id[end_id] = end_el

    for owner_uri, prop_el in pending_attributes:
        prop_uri = _property_uri_from_xml(owner_uri, prop_el)
        meta = _parse_semantic_comment(_comment_body(prop_el))
        prop_name = _attr(prop_el, "name") or local_name(prop_uri)

        prop_label = meta.get("label") or prop_name
        prop_definition = meta.get("definition")
        prop_usage = meta.get("usage_note")

        range_uri, is_primitive = _resolve_type_from_owned_attribute(prop_el, classifier_uri_by_id)

        g.remove((prop_uri, RDF.type, OWL.DatatypeProperty))
        g.remove((prop_uri, RDF.type, OWL.ObjectProperty))
        g.add((prop_uri, RDF.type, OWL.DatatypeProperty if is_primitive else OWL.ObjectProperty))

        _set_literal(g, prop_uri, RDFS.label, prop_label, lang="fr")
        _set_literal(g, prop_uri, RDFS.comment, prop_definition, lang="fr")
        _set_literal(g, prop_uri, SKOS.scopeNote, prop_usage, lang="fr")
        _add_uri(g, prop_uri, RDFS.domain, owner_uri)

        if range_uri:
            _set_uri(g, prop_uri, RDFS.range, range_uri)

        lb, ub = _parse_multiplicity_bounds(prop_el)
        _set_literal_if_absent(g, prop_uri, UML_META.lowerBound, lb)
        _set_literal_if_absent(g, prop_uri, UML_META.upperBound, ub)

    for source_uri, gen_el in pending_generalizations:
        target_id = _attr(gen_el, "general")
        target_uri = classifier_uri_by_id.get(target_id)
        if target_uri is not None:
            g.add((source_uri, RDFS.subClassOf, target_uri))

    for assoc_el in pending_associations:
        assoc_meta = _parse_semantic_comment(_comment_body(assoc_el))
        assoc_name = _attr(assoc_el, "name")

        ends = _find_direct_children(assoc_el, "ownedEnd")
        if len(ends) < 2:
            member_end_refs = [tok for tok in _attr(assoc_el, "memberEnd").split() if tok]
            resolved_ends: list[ET.Element] = []
            for ref in member_end_refs:
                end_el = association_end_by_id.get(ref)
                if end_el is not None:
                    resolved_ends.append(end_el)
            ends = resolved_ends

        if len(ends) < 2:
            continue

        source_end = ends[0]
        target_end = ends[1]

        source_id = _resolve_end_type_id(source_end)
        target_id = _resolve_end_type_id(target_end)

        if source_id is None or target_id is None or source_id == "" or target_id == "":
            continue

        source_uri = classifier_uri_by_id.get(source_id)
        target_uri = classifier_uri_by_id.get(target_id)

        if source_uri is None or target_uri is None:
            continue

        semantic_uri = (
            _norm_text(assoc_meta.get("uri"))
            or f"urn:relation:{_slugify_uri(assoc_name or 'relation')}"
        )
        prop_uri = URIRef(semantic_uri)

        rel_label = assoc_meta.get("label") or assoc_name or local_name(prop_uri)
        rel_definition = assoc_meta.get("definition")
        rel_usage = assoc_meta.get("usage_note")

        g.remove((prop_uri, RDF.type, OWL.DatatypeProperty))
        g.add((prop_uri, RDF.type, OWL.ObjectProperty))

        _set_literal(g, prop_uri, RDFS.label, rel_label, lang="fr")
        _set_literal(g, prop_uri, RDFS.comment, rel_definition, lang="fr")
        _set_literal(g, prop_uri, SKOS.scopeNote, rel_usage, lang="fr")
        _add_uri(g, prop_uri, RDFS.domain, source_uri)
        _add_uri(g, prop_uri, RDFS.range, target_uri)

        relationship = "Association"
        target_aggregation = _attr(target_end, "aggregation")
        source_aggregation = _attr(source_end, "aggregation")

        aggregation_value = target_aggregation or source_aggregation
        if aggregation_value == "shared":
            relationship = "Aggregation"
        elif aggregation_value == "composite":
            relationship = "Composition"

        _set_literal(g, prop_uri, UML_META.relationshipType, relationship)

        lb = _compose_multiplicity(*_parse_multiplicity_bounds(source_end))
        rb = _compose_multiplicity(*_parse_multiplicity_bounds(target_end))
        lt = _attr(source_end, "name")
        rt = _attr(target_end, "name")

        _set_literal_if_absent(g, prop_uri, UML_META.leftMultiplicity, lb)
        _set_literal_if_absent(g, prop_uri, UML_META.rightMultiplicity, rb)
        _set_literal_if_absent(g, prop_uri, UML_META.leftRole, lt)
        _set_literal_if_absent(g, prop_uri, UML_META.rightRole, rt)

    return g


def graph_from_model(model: dict[str, Any], user: str = "", name: str = "") -> Graph:
    print(f"[DEBUG graph_from_model] Chargement du graphe RDF à partir du modèle...")
    
    # PRIORITÉ 1 : Si nous avons du Turtle brut (ttl_raw), c'est la vérité sémantique absolue et la plus à jour.
    ttl_raw = _norm_text(model.get("ttl_raw"))
    if ttl_raw:
        print(f"[DEBUG graph_from_model] Source détectée : Turtle brut (Longueur={len(ttl_raw)}). Construction directe.")
        g = Graph()
        try:
            g.parse(data=ttl_raw, format="turtle")
            print(f"[DEBUG graph_from_model] Graphe RDF parsé avec succès depuis ttl_raw. {len(g)} triplets chargés.")
            return g
        except Exception as e:
            print(f"[DEBUG graph_from_model] ERREUR critique lors du parsing Turtle : {e}")
            raise e

    # PRIORITÉ 2 : Si nous avons déjà des éléments et connecteurs pré-remplis dans le modèle (ex: importés via un parseur EA dédié),
    # nous devons absolument construire le graphe RDF depuis ces derniers !
    xmi = _xmi_view(model)
    if xmi.get("elements"):
        print(f"[DEBUG graph_from_model] Source détectée : Éléments/Connecteurs pré-existants. Construction du graphe depuis les dictionnaires.")
        temp_model = dict(model)
        for key in ("xmi_raw", "xmi_xml", "xmiraw", "xmixml"):
            temp_model.pop(key, None)
        return build_graph_from_xmi_model(
            temp_model,
            user=user,
            name=name or _norm_text(model.get("name")) or "Generated",
        )

    # PRIORITÉ 3 : Si on n'a ni ttl_raw, ni éléments UML, mais qu'on a du XMI brut, on le parse.
    raw_xmi = _norm_text(model.get("xmi_raw")) or _norm_text(model.get("xmi_xml"))
    if raw_xmi:
        print(f"[DEBUG graph_from_model] Source détectée : XMI brut d'importation (sans éléments pré-existants).")
        return build_graph_from_xmi_model(
            model,
            user=user,
            name=name or _norm_text(model.get("name")) or "Generated",
        )

    print(f"[DEBUG graph_from_model] Aucune source détectée, création d'un nouveau graphe sémantique vierge.")
    return _new_graph(user, name or "Generated")


def upload_model(
    model: dict[str, Any],
    user: str = "",
    name: str = "",
) -> dict[str, Any]:
    """
    Sauvegarde le modèle d'import brut puis le synchronise immédiatement
    afin d'injecter la clé `ttl_raw` permettant un export sans délai,
    tout en préservant le fichier source EA (`keep_raw=True`).
    """
    print(f"[DEBUG upload_model] Sauvegarde et synchronisation initiale pour user='{user}', name='{name}'")
    folder_path = MODELS_PATH / user
    folder_path.mkdir(parents=True, exist_ok=True)
    fp = _find_file(user, name)

    with open(fp, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)

    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Déclenche la synchronisation immédiate qui greffe le `ttl_raw`
    # tout en conservant le `xmi_raw` intact (keep_raw=True) et les éléments existants.
    data = _ensure_synchronized(data, user, name, keep_raw=True)

    return data


def load_full_model(user: str = "", name: str = "") -> dict[str, Any]:
    fp = ensure_model_exists(user=user, name=name)
    model = _load_model(fp)
    return _ensure_synchronized(model, user, name)


def get_model(user: str = "", name: str = "") -> dict[str, Any]:
    fp = ensure_model_exists(user=user, name=name)
    model = _load_model(fp)
    return _ensure_synchronized(model, user, name)


def _iter_referenceable_class_uris(g: Graph):
    seen: set[str] = set()

    def _yield_if_uri(node):
        if isinstance(node, URIRef):
            s = str(node)
            if s not in seen and primitive_from_range(node) is None:
                seen.add(s)
                yield node

    for cls in g.subjects(RDF.type, OWL.Class):
        yield from _yield_if_uri(cls)

    for _, _, domain in g.triples((None, RDFS.domain, None)):
        yield from _yield_if_uri(domain)

    for _, _, rng in g.triples((None, RDFS.range, None)):
        yield from _yield_if_uri(rng)

    for child, parent in g.subject_objects(RDFS.subClassOf):
        yield from _yield_if_uri(child)
        yield from _yield_if_uri(parent)


def _is_referenceable_class_uri(g: Graph, uri: URIRef) -> bool:
    if primitive_from_range(uri) is not None:
        return False

    if (uri, RDF.type, OWL.Class) in g:
        return True

    if (None, RDFS.domain, uri) in g:
        return True

    if (None, RDFS.range, uri) in g:
        return True

    if (uri, RDFS.subClassOf, None) in g:
        return True

    if (None, RDFS.subClassOf, uri) in g:
        return True

    return False


def find_class_by_label(g: Graph, class_name: str) -> URIRef | None:
    wanted = _canonical_text(class_name)
    if not wanted:
        return None

    if "://" in class_name or class_name.startswith("urn:"):
        uri = URIRef(class_name)
        if _is_referenceable_class_uri(g, uri):
            return uri

    for s in _iter_referenceable_class_uris(g):
        if _canonical_text(local_name(s)) == wanted:
            return s

        label = get_label(g, s)
        if label and _canonical_text(label) == wanted:
            return s

        for raw_label in g.objects(s, RDFS.label):
            if _canonical_text(str(raw_label)) == wanted:
                return s

    return None


def find_class_by_label_or_xmi(
        g: Graph, class_name: str, package_name: str | None = None) -> URIRef | None:
    found = find_class_by_label(g, class_name)
    if found:
        return found

    wanted = _canonical_text(class_name)
    if not wanted:
        return None

    xmi = build_model(g, package_name=package_name)
    for el in xmi.get("elements", []):
        if el.get("type") != "uml:Class":
            continue
        if _canonical_text(el.get("name")) != wanted:
            continue

        for tag in el.get("tags", []):
            if tag.get("name") == "uri" and tag.get("value"):
                return URIRef(tag["value"])

    return None


def _find_class_result(
    model: dict[str, Any],
    title: str | None = None,
    uri: str | None = None,
) -> dict[str, Any]:
    for el in _xmi_view(model).get("elements", []):
        if el.get("type") != "uml:Class":
            continue
        if uri:
            if el.get("uri") == uri:
                return el
            for tag in el.get("tags", []):
                if tag.get("name") == "uri" and tag.get("value") == uri:
                    return el
        if title and _canonical_text(el.get("name")) == _canonical_text(title):
            return el
    return {}


def _find_attribute_result(
    model: dict[str, Any],
    class_name: str,
    attr_uri: str,
    class_uri: str = "",
) -> dict[str, Any]:
    for el in _xmi_view(model).get("elements", []):
        if el.get("type") != "uml:Class":
            continue

        name_match = _canonical_text(el.get("name")) == _canonical_text(class_name)
        uri_match = class_uri and (
            el.get("uri") == class_uri or
            any(tag.get("name") == "uri" and tag.get("value") == class_uri for tag in el.get("tags", []))
        )

        if not name_match and not uri_match:
            continue

        for attr in el.get("attributes", []):
            if attr.get("uri") == attr_uri or any(tag.get("name") == "uri" and tag.get("value") == attr_uri for tag in attr.get("tags_attribute", [])):
                res = dict(attr)
                res["class_uri"] = el.get("uri")
                res["class_id"] = el.get("ID")
                res["class_name"] = el.get("name")
                return res
    return {}


def _find_connector_result(
    model: dict[str, Any],
    rel_uri: str = "",
    relationship: str = "",
    source_name: str = "",
    target_name: str = "",
    rel_label: str = "",
    connector_id: str = "",
) -> dict[str, Any]:
    for conn in _xmi_view(model).get("connectors", []):
        if connector_id and _connector_id_from_view(conn) == connector_id:
            return conn

        semantic_uri = _semantic_uri_from_connector_view(conn)
        if relationship == "Generalization":
            semantic_match = True
        else:
            semantic_match = (not rel_uri) or semantic_uri == rel_uri

        if (
            semantic_match
            and ((not relationship) or _safe_xmi_value(conn.get("relationship")) == relationship)
            and ((not source_name)
                 or _canonical_text(conn.get("source_name")) == _canonical_text(source_name))
            and ((not target_name)
                 or _canonical_text(conn.get("target_name")) == _canonical_text(target_name))
            and ((not rel_label) or _canonical_text(conn.get("name")) == _canonical_text(rel_label))
        ):
            return conn

    return {}


def add_class(
    title: str,
    definition: str,
    usage_note: str,
    user: str = "",
    name: str = "",
    package: str | None = None,
    uri: str | None = None,
    file_name: str = "",
) -> dict[str, Any]:
    print(f"[DEBUG add_class] Début add_class : title='{title}', uri='{uri}', user='{user}', name='{name}', file_name='{file_name}'")
    package = package or ""

    fp = ensure_model_exists(user=user, name=file_name or name)
    model = _load_model(fp)
    model = _ensure_synchronized(model, user, name)
    g = graph_from_model(model, user=user, name=name or "Generated")

    existing = find_class_by_label(g, title)
    print(f"[DEBUG add_class] Recherche de classe existante pour '{title}'... Trouvée : {existing}")
    if existing:
        print(f"[DEBUG add_class] La classe '{title}' existe déjà sous l'URI sémantique {existing}. Synchronisation et sauvegarde.")
        model = _sync_model_from_graph(
            g,
            model,
            package_name=name or "Generated",
            source_format=_default_source_format(model),
            keep_raw=False,
            update_elements=True,
        )
        _save_model(fp, model, user=user, name=file_name or name)
        return _find_class_result(model, title=title) or {
            "error": f"La classe '{title}' existe déjà."
        }

    raw_uri = _norm_text(uri)
    if raw_uri and (
        raw_uri.startswith("http://")
        or raw_uri.startswith("https://")
        or raw_uri.startswith("urn:")
    ):
        class_uri = URIRef(raw_uri)
    else:
        class_uri = URIRef(f"urn:class:{_slugify_uri(title)}")

    print(f"[DEBUG add_class] Enregistrement de la nouvelle classe avec l'URI : {class_uri}")
    g.add((class_uri, RDF.type, OWL.Class))
    _set_literal(g, class_uri, RDFS.label, title, lang="fr")
    _set_literal(g, class_uri, RDFS.comment, definition, lang="fr")
    _set_literal(g, class_uri, SKOS.scopeNote, usage_note, lang="fr")

    model = _sync_model_from_graph(
        g,
        model,
        package_name=name or "Generated",
        source_format=_default_source_format(model),
        keep_raw=False,
        update_elements=True,
    )
    model = _save_model(fp, model, user=user, name=file_name or name)

    result = _find_class_result(model, title=title)
    print(f"[DEBUG add_class] Résultat renvoyé après ajout : {bool(result)}")
    return result or {
        "error": f"Classe '{title}' créée mais introuvable dans le résultat."
    }


def add_attribute(
    class_name: str,
    attr_label: str,
    attr_definition: str,
    attr_uri: str,
    attr_usage_note: str = "",
    attr_type: str | None = "",
    lower_bounds: str = "",
    upper_bounds: str = "",
    user: str = "",
    name: str = "",
    file_name: str = "",
) -> dict[str, Any]:
    print(f"[DEBUG add_attribute] Début add_attribute : class_name='{class_name}', attr_label='{attr_label}', attr_uri='{attr_uri}', file_name='{file_name}'")
    fp = ensure_model_exists(user=user, name=file_name or name)
    model = _load_model(fp)
    model = _ensure_synchronized(model, user, name)
    g = graph_from_model(model, user=user, name=name or "Generated")

    class_uri = find_class_by_label(g, class_name)
    print(f"[DEBUG add_attribute] Recherche de la classe parent '{class_name}'... Trouvée : {class_uri}")
    if not class_uri:
        print(f"[DEBUG add_attribute] ERREUR: La classe parent '{class_name}' est introuvable!")
        all_classes_labels = [get_label(g, c) or local_name(c) for c in g.subjects(RDF.type, OWL.Class)]
        print(f"[DEBUG add_attribute] Classes actuellement disponibles dans le graphe : {all_classes_labels}")
        return {"error": f"Classe source introuvable : '{class_name}'"}

    prop_uri = URIRef(attr_uri)
    print(f"[DEBUG add_attribute] Ajout des triplets pour l'attribut URI: {prop_uri}")
    _set_literal(g, prop_uri, RDFS.label, attr_label, lang="fr")
    _set_literal(g, prop_uri, RDFS.comment, attr_definition, lang="fr")
    _set_literal(g, prop_uri, SKOS.scopeNote, attr_usage_note, lang="fr")
    _add_uri(g, prop_uri, RDFS.domain, class_uri)

    _set_literal(g, prop_uri, UML_META.lowerBound, lower_bounds)
    _set_literal(g, prop_uri, UML_META.upperBound, upper_bounds)

    canonical_range_uri, is_primitive = resolve_type_uri(attr_type)
    print(f"[DEBUG add_attribute] Type d'attribut résolu : range='{canonical_range_uri}', is_primitive={is_primitive}")

    g.remove((prop_uri, RDF.type, OWL.DatatypeProperty))
    g.remove((prop_uri, RDF.type, OWL.ObjectProperty))
    g.remove((prop_uri, RDFS.range, None))

    if is_primitive:
        g.add((prop_uri, RDF.type, OWL.DatatypeProperty))
        _set_uri(g, prop_uri, RDFS.range, canonical_range_uri)
    else:
        g.add((prop_uri, RDF.type, OWL.ObjectProperty))
        target_uri = find_class_by_label(g, attr_type)
        if not target_uri:
            attr_type = attr_type or ""
            if "://" in attr_type or attr_type.startswith("urn:"):
                target_uri = URIRef(attr_type)
            else:
                target_uri = URIRef(f"urn:class:{_slugify_uri(attr_type)}")
            g.add((target_uri, RDF.type, OWL.Class))
            _set_literal(g, target_uri, RDFS.label, attr_type, lang="fr")

        _set_uri(g, prop_uri, RDFS.range, target_uri)

    model = _sync_model_from_graph(
        g,
        model,
        package_name=name or "Generated",
        source_format=_default_source_format(model),
        keep_raw=False,
        update_elements=True,
    )
    model = _save_model(fp, model, user=user, name=file_name or name)

    result = _find_attribute_result(
        model,
        class_name=class_name,
        attr_uri=attr_uri,
        class_uri=str(class_uri),
    )
    print(f"[DEBUG add_attribute] Résultat renvoyé après ajout : {bool(result)}")
    return result or {
        "error": f"Attribut '{attr_label}' créé mais introuvable dans le résultat."
    }


def add_connector(
    source_name: str,
    target_name: str,
    rel_label: str,
    rel_definition: str,
    rel_uri: str,
    relationship: str,
    lb: str = "",
    rb: str = "",
    lt: str = "",
    rt: str = "",
    rel_usage_note: str = "",
    user: str = "",
    name: str = "",
    file_name: str = "",
) -> dict[str, Any]:
    print(f"[DEBUG add_connector] Début add_connector : source='{source_name}', target='{target_name}', label='{rel_label}', type='{relationship}', file_name='{file_name}'")
    fp = ensure_model_exists(user=user, name=file_name or name)
    model = _load_model(fp)
    model = _ensure_synchronized(model, user, name)
    g = graph_from_model(model, user=user, name=name or "Generated")

    source_uri = find_class_by_label_or_xmi(g, source_name, package_name=name or "Generated")
    print(f"[DEBUG add_connector] Résolution de la classe source sémantique : {source_uri}")
    if not source_uri:
        print(f"[DEBUG add_connector] ERREUR: Classe source '{source_name}' introuvable!")
        return {"error": f"Classe source introuvable : '{source_name}'"}

    target_uri = find_class_by_label_or_xmi(g, target_name, package_name=name or "Generated")
    print(f"[DEBUG add_connector] Résolution de la classe cible sémantique : {target_uri}")
    if not target_uri:
        print(f"[DEBUG add_connector] ERREUR: Classe cible '{target_name}' introuvable!")
        return {"error": f"Classe cible introuvable : '{target_name}'"}

    relationship = _norm_text(relationship) or "Association"
    lb = _safe_xmi_value(lb)
    rb = _safe_xmi_value(rb)
    lt = _safe_xmi_value(lt)
    rt = _safe_xmi_value(rt)
    rel_uri = _safe_xmi_value(rel_uri)

    connector_semantic_uri = "" if relationship == "Generalization" else rel_uri
    connector_id = _build_connector_id(
        connector_semantic_uri,
        relationship,
        str(source_uri),
        str(target_uri),
    )
    print(f"[DEBUG add_connector] Enregistrement du connecteur ID={connector_id}")

    if relationship == "Generalization":
        g.add((source_uri, RDFS.subClassOf, target_uri))
    else:
        prop_uri = URIRef(rel_uri)
        g.remove((prop_uri, RDF.type, OWL.DatatypeProperty))
        g.add((prop_uri, RDF.type, OWL.ObjectProperty))

        _set_literal(g, prop_uri, RDFS.label, rel_label, lang="fr")
        _set_literal(g, prop_uri, RDFS.comment, rel_definition, lang="fr")
        _set_literal(g, prop_uri, SKOS.scopeNote, rel_usage_note, lang="fr")
        _add_uri(g, prop_uri, RDFS.domain, source_uri)
        _add_uri(g, prop_uri, RDFS.range, target_uri)
        _set_literal(g, prop_uri, UML_META.relationshipType, relationship)
        
        # Insertion des cardinalités dans le graphe sémantique RDF
        _set_literal(g, prop_uri, UML_META.leftMultiplicity, lb)
        _set_literal(g, prop_uri, UML_META.rightMultiplicity, rb)
        _set_literal(g, prop_uri, UML_META.leftRole, lt)
        _set_literal(g, prop_uri, UML_META.rightRole, rt)

    connector_overrides = _extract_connector_overrides(model)
    override_key = _connector_key_from_parts(
        connector_semantic_uri,
        relationship,
        str(source_uri),
        str(target_uri),
    )
    connector_overrides[override_key] = {
        "connector_id": connector_id,
        "semantic_uri": connector_semantic_uri,
        "relationship": relationship,
        "name": rel_label or ("subClassOf" if relationship == "Generalization"
                              else local_name(rel_uri)),
        "lb": lb,
        "lt": lt,
        "rb": rb,
        "rt": rt,
        "source_name": source_name,
        "target_name": target_name,
    }

    model = _sync_model_from_graph(
        g,
        model,
        package_name=name or "Generated",
        source_format=_default_source_format(model),
        connector_overrides=connector_overrides,
        keep_raw=False,
        update_elements=True,
    )
    model = _save_model(fp, model, user=user, name=file_name or name)

    result = _find_connector_result(
        model,
        rel_uri=rel_uri,
        relationship=relationship,
        source_name=source_name,
        target_name=target_name,
        rel_label=rel_label,
        connector_id=connector_id,
    )
    print(f"[DEBUG add_connector] Connecteur recherché dans le résultat : {bool(result)}")
    return result or {
        "error": f"Connecteur '{rel_label}' créé mais introuvable dans le résultat."
    }


def _remove_none_deep(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _remove_none_deep(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_remove_none_deep(v) for v in value if v is not None]
    return value


def build_xmi_bytes(model: dict[str, Any]) -> bytes:
    xmi = _remove_none_deep(model.get("xmi", {}))
    elements = xmi.get("elements", []) or []
    connectors = xmi.get("connectors", []) or []

    NS_XMI = "http://schema.omg.org/spec/XMI/2.1"
    NS_UML = "http://schema.omg.org/spec/UML/2.1"

    from xml.etree.ElementTree import register_namespace

    register_namespace("xmi", NS_XMI)
    register_namespace("uml", NS_UML)

    def qname(ns: str, tag: str) -> str:
        return f"{{{ns}}}{tag}"

    def text(value: Any) -> str:
        return _safe_xmi_value(value)

    def first_tag_value(tags: list[dict[str, Any]] | None, key: str) -> str:
        return _first_tag_value(tags, key)

    def add_comment(parent: Element, body_text: str | None, seed: str) -> None:
        body_text = _norm_text(body_text)
        if not body_text:
            return
        owned_comment = SubElement(parent, "ownedComment")
        owned_comment.set(qname(NS_XMI, "id"), ea_id("COMMENT", seed))
        owned_comment.set(qname(NS_XMI, "type"), "uml:Comment")
        body = SubElement(owned_comment, "body")
        body.text = body_text

    def build_semantic_comment_from_tags(tags: list[dict[str, Any]] | None) -> str | None:
        if not tags:
            return None

        parts: list[str] = []

        semantic_uri = first_tag_value(tags, "semantic_uri")
        uri = first_tag_value(tags, "uri")
        label = first_tag_value(tags, "label-fr") or first_tag_value(tags, "label")
        definition = first_tag_value(tags, "definition-fr") or first_tag_value(tags, "definition")
        usage_note = first_tag_value(tags, "usageNote-fr") or first_tag_value(tags, "usageNote")
        referenced = first_tag_value(tags, "referenced")
        connector_id = first_tag_value(tags, "connector_id")

        if semantic_uri:
            parts.append(f"URI: {semantic_uri}")
        elif uri:
            parts.append(f"URI: {uri}")

        if label:
            parts.append(f"Label: {label}")
        if definition:
            parts.append(f"Definition: {definition}")
        if usage_note:
            parts.append(f"Usage note: {usage_note}")
        if referenced:
            parts.append(f"Referenced: {referenced}")
        if connector_id:
            parts.append(f"Connector ID: {connector_id}")

        return "\n".join(parts) if parts else None

    def _normalize_mult_token(token: str | None) -> str | None:
        token = _norm_text(token)
        if not token:
            return None

        t = token.strip().lower()

        if t in {"n", "m", "many", "*"}:
            return "*"

        if t.isdigit():
            return t

        return token.strip()

    def parse_multiplicity(raw: Any) -> tuple[str | None, str | None]:
        raw = text(raw).strip()
        if not raw:
            return (None, None)

        normalized = (
            raw.replace(" ", "")
            .replace("many", "*")
            .replace("Many", "*")
            .replace("MANY", "*")
            .replace("N", "*")
            .replace("n", "*")
            .replace("M", "*")
        )

        if normalized == "*":
            return ("0", "*")

        if ".." in normalized:
            lower, upper = normalized.split("..", 1)
            lower = _normalize_mult_token(lower)
            upper = _normalize_mult_token(upper)

            if lower == "*":
                lower = "0"
            if upper == "*" and lower is None:
                lower = "0"

            return (lower, upper)

        one = _normalize_mult_token(normalized)
        if one == "*":
            return ("0", "*")

        return (one, one)

    def normalize_bounds(lower_raw: Any, upper_raw: Any) -> tuple[str | None, str | None]:
        lower = _normalize_mult_token(_norm_text(lower_raw))
        upper = _normalize_mult_token(_norm_text(upper_raw))

        if lower == "*":
            lower = "0"
        if upper == "*" and lower is None:
            lower = "0"

        return (lower, upper)

    def add_lower_upper(owner: Element, lower: str | None, upper: str | None, seed: str) -> None:
        if lower is not None:
            safe_lower = "0" if lower == "*" else lower
            if safe_lower.isdigit():
                lower_value = SubElement(owner, "lowerValue")
                lower_value.set(qname(NS_XMI, "type"), "uml:LiteralInteger")
                lower_value.set(qname(NS_XMI, "id"), ea_id("LOW", f"{seed}:lower"))
                lower_value.set("value", safe_lower)

        if upper is not None:
            if upper == "*":
                upper_value = SubElement(owner, "upperValue")
                upper_value.set(qname(NS_XMI, "type"), "uml:LiteralUnlimitedNatural")
                upper_value.set(qname(NS_XMI, "id"), ea_id("UP", f"{seed}:upper"))
                upper_value.set("value", "*")
            elif upper.isdigit():
                upper_value = SubElement(owner, "upperValue")
                upper_value.set(qname(NS_XMI, "type"), "uml:LiteralInteger")
                upper_value.set(qname(NS_XMI, "id"), ea_id("UP", f"{seed}:upper"))
                upper_value.set("value", upper)

    def primitive_href(attr_type: str) -> str | None:
        canonical_range_uri, is_primitive = resolve_type_uri(attr_type)
        if not canonical_range_uri or not is_primitive:
            return None

        primitive_name = primitive_from_range(URIRef(canonical_range_uri))
        if not primitive_name:
            return None

        primitive_map = {
            "String": "String",
            "Boolean": "Boolean",
            "Integer": "Integer",
            "Real": "Real",
            "Date": "Date",
            "DateTime": "DateTime",
            "Time": "Time",
            "URI": "URI",
        }

        mapped = primitive_map.get(primitive_name)
        if not mapped:
            return None

        return f"http://www.omg.org/spec/UML/20131001/PrimitiveTypes.xmi#{mapped}"

    def is_connector_shadow_attribute(attr: dict[str, Any] | None) -> bool:
        if not isinstance(attr, dict):
            return False

        attr_id = text(attr.get("ID"))
        if attr_id.startswith("CONN_"):
            return True

        tags = attr.get("tags_attribute", []) or []
        if first_tag_value(tags, "connector_id"):
            return True
        if first_tag_value(tags, "semantic_uri"):
            return True

        uri = first_tag_value(tags, "uri")
        if uri and (uri.startswith("http://") or uri.startswith("https://") or uri.startswith("urn:")):
            for conn in connectors:
                conn_semantic_uri = _semantic_uri_from_connector_view(conn)
                if conn_semantic_uri and conn_semantic_uri == uri:
                    return True

        return False

    def connector_comment_from_conn(conn: dict[str, Any]) -> str | None:
        merged_rel_tags = (
            (conn.get("tags_target") or [])
            + (conn.get("tags_source") or [])
            + (conn.get("tags") or [])
        )

        rel_comment = build_semantic_comment_from_tags(merged_rel_tags)
        if rel_comment:
            return rel_comment

        semantic_uri = _semantic_uri_from_connector_view(conn)
        rel_name = text(conn.get("name"))
        rel_definition = (
            first_tag_value(merged_rel_tags, "definition-fr")
            or first_tag_value(merged_rel_tags, "definition")
        )
        rel_usage = (
            first_tag_value(merged_rel_tags, "usageNote-fr")
            or first_tag_value(merged_rel_tags, "usageNote")
        )

        rel_parts = []
        if semantic_uri:
            rel_parts.append(f"URI: {semantic_uri}")
        if rel_name:
            rel_parts.append(f"Label: {rel_name}")
        if rel_definition:
            rel_parts.append(f"Definition: {rel_definition}")
        if rel_usage:
            rel_parts.append(f"Usage note: {rel_usage}")

        connector_id = _connector_id_from_view(conn)
        if connector_id:
            rel_parts.insert(0, f"Connector ID: {connector_id}")

        return "\n".join(rel_parts) if rel_parts else None

    root = Element(qname(NS_XMI, "XMI"))
    root.set(qname(NS_XMI, "version"), "2.1")

    documentation = SubElement(root, qname(NS_XMI, "Documentation"))
    documentation.set("exporter", "HarmonIA")
    documentation.set("exporterVersion", "2.1")

    model_name = text(model.get("name") or "model")
    model_id = ea_id("MODEL", model_name or "model")

    uml_model = SubElement(root, qname(NS_UML, "Model"))
    uml_model.set(qname(NS_XMI, "id"), model_id)
    uml_model.set(qname(NS_XMI, "type"), "uml:Model")
    uml_model.set("name", model_name or "model")
    uml_model.set("visibility", "public")

    element_by_id: dict[str, dict[str, Any]] = {}
    element_xml_by_id: dict[str, Element] = {}
    package_ids: set[str] = set()
    classifier_ids_by_name: dict[str, str] = {}
    classifier_ids_by_canonical_name: dict[str, str] = {}
    classifier_ids_by_uri: dict[str, str] = {}

    for el in elements:
        el_id = text(el.get("ID"))
        el_type = text(el.get("type"))
        if not el_id:
            continue

        element_by_id[el_id] = el

        if el_type == "uml:Package":
            package_ids.add(el_id)

        if el_type in {"uml:Class", "uml:DataType", "uml:Enumeration"}:
            el_name = text(el.get("name"))
            if el_name:
                classifier_ids_by_name[el_name] = el_id
                classifier_ids_by_canonical_name[_canonical_text(el_name)] = el_id

            uri = first_tag_value(el.get("tags", []), "uri")
            if uri:
                classifier_ids_by_uri[uri] = el_id

    def resolve_parent_container(el: dict[str, Any]) -> Element:
        parent_id = text(el.get("package"))
        if parent_id and parent_id in package_ids and parent_id in element_xml_by_id:
            return element_xml_by_id[parent_id]
        return uml_model

    def create_package(el: dict[str, Any], parent_xml: Element) -> Element:
        el_id = text(el.get("ID"))
        pkg_el = SubElement(parent_xml, "packagedElement")
        pkg_el.set(qname(NS_XMI, "id"), el_id)
        pkg_el.set(qname(NS_XMI, "type"), "uml:Package")
        pkg_el.set("name", text(el.get("name")) or "Package")
        pkg_el.set("visibility", "public")
        element_xml_by_id[el_id] = pkg_el

        add_comment(
            pkg_el,
            build_semantic_comment_from_tags(el.get("tags", [])),
            f"{el_id}:comment",
        )
        return pkg_el

    def create_classifier(el: dict[str, Any], parent_xml: Element) -> Element:
        el_id = text(el.get("ID"))
        el_type = text(el.get("type"))
        cls_el = SubElement(parent_xml, "packagedElement")
        cls_el.set(qname(NS_XMI, "id"), el_id)
        cls_el.set(qname(NS_XMI, "type"), el_type)
        cls_el.set("name", text(el.get("name")))
        cls_el.set("visibility", "public")
        element_xml_by_id[el_id] = cls_el

        add_comment(
            cls_el,
            build_semantic_comment_from_tags(el.get("tags", [])),
            f"{el_id}:comment",
        )

        if el_type in {"uml:Class", "uml:DataType"}:
            for attr in el.get("attributes", []) or []:
                if is_connector_shadow_attribute(attr):
                    continue

                attr_name = text(attr.get("name")) or "attribute"
                attr_type = text(attr.get("type"))
                attr_id = text(attr.get("ID")) or ea_id("ATTR", f"{el_id}:{attr_name}")

                owned_attr = SubElement(cls_el, "ownedAttribute")
                owned_attr.set(qname(NS_XMI, "id"), attr_id)
                owned_attr.set(qname(NS_XMI, "type"), "uml:Property")
                owned_attr.set("name", attr_name)
                owned_attr.set("visibility", "public")

                href = primitive_href(attr_type)
                if href:
                    type_el = SubElement(owned_attr, "type")
                    type_el.set(qname(NS_XMI, "type"), "uml:PrimitiveType")
                    type_el.set("href", href)
                else:
                    target_id = (
                        classifier_ids_by_name.get(attr_type)
                        or classifier_ids_by_canonical_name.get(_canonical_text(attr_type))
                        or classifier_ids_by_uri.get(attr_type)
                    )
                    if target_id:
                        owned_attr.set("type", target_id)
                    elif attr_type:
                        add_comment(owned_attr, f"type: {attr_type}", f"{attr_id}:type_fallback")

                lower, upper = normalize_bounds(
                    attr.get("lower_bounds"),
                    attr.get("upper_bounds"),
                )
                if lower is not None or upper is not None:
                    add_lower_upper(owned_attr, lower, upper, attr_id)

                add_comment(
                    owned_attr,
                    build_semantic_comment_from_tags(attr.get("tags_attribute", [])),
                    f"{attr_id}:comment",
                )

        elif el_type == "uml:Enumeration":
            categories = el.get("categories", []) or []
            if not categories and el.get("attributes"):
                categories = [
                    text(a.get("name"))
                    for a in (el.get("attributes") or [])
                    if text(a.get("name"))
                ]

            for i, literal_name in enumerate(categories, start=1):
                lit = SubElement(cls_el, "ownedLiteral")
                lit.set(qname(NS_XMI, "id"), ea_id("LIT", f"{el_id}:{i}:{literal_name}"))
                lit.set(qname(NS_XMI, "type"), "uml:EnumerationLiteral")
                lit.set("name", text(literal_name))

        return cls_el

    unresolved = [el for el in elements if text(el.get("ID"))]
    max_passes = max(3, len(unresolved) + 2)
    pass_count = 0

    while unresolved and pass_count < max_passes:
        pass_count += 1
        next_round: list[dict[str, Any]] = []

        for el in unresolved:
            el_type = text(el.get("type"))
            parent_id = text(el.get("package"))

            if parent_id and parent_id in package_ids and parent_id not in element_xml_by_id:
                next_round.append(el)
                continue

            parent_xml = resolve_parent_container(el)

            if el_type == "uml:Package":
                create_package(el, parent_xml)
            elif el_type in {"uml:Class", "uml:DataType", "uml:Enumeration"}:
                create_classifier(el, parent_xml)

        if len(next_round) == len(unresolved):
            for el in next_round:
                el_type = text(el.get("type"))
                if el_type == "uml:Package":
                    create_package(el, uml_model)
                elif el_type in {"uml:Class", "uml:DataType", "uml:Enumeration"}:
                    create_classifier(el, uml_model)
            break

        unresolved = next_round

    for conn in connectors:
        relationship = text(conn.get("relationship")) or "Association"
        relationship_lower = relationship.lower()

        source_id = text(conn.get("source_id"))
        target_id = text(conn.get("target_id"))

        if not source_id and text(conn.get("source_name")):
            source_id = (
                classifier_ids_by_name.get(text(conn.get("source_name")))
                or classifier_ids_by_canonical_name.get(_canonical_text(text(conn.get("source_name"))))
                or ""
            )

        if not target_id and text(conn.get("target_name")):
            target_id = (
                classifier_ids_by_name.get(text(conn.get("target_name")))
                or classifier_ids_by_canonical_name.get(_canonical_text(text(conn.get("target_name"))))
                or ""
            )

        if not source_id or not target_id:
            continue

        connector_id = _connector_id_from_view(conn) or ea_id(
            "CONN",
            f"{source_id}:{target_id}:{text(conn.get('name'))}:{relationship}",
        )

        if relationship_lower == "generalization":
            source_cls = element_xml_by_id.get(source_id)
            if source_cls is None:
                continue

            gen = SubElement(source_cls, "generalization")
            gen.set(qname(NS_XMI, "id"), connector_id)
            gen.set(qname(NS_XMI, "type"), "uml:Generalization")
            gen.set("general", target_id)

            add_comment(gen, connector_comment_from_conn(conn), f"{connector_id}:comment")
            continue

        source_el = element_by_id.get(source_id, {})
        assoc_parent = resolve_parent_container(source_el) if source_el else uml_model

        assoc = SubElement(assoc_parent, "packagedElement")
        assoc.set(qname(NS_XMI, "id"), connector_id)
        assoc.set(qname(NS_XMI, "type"), "uml:Association")
        assoc.set("name", text(conn.get("name")))
        assoc.set("visibility", "public")

        end1_id = ea_id("END", f"{connector_id}:source")
        end2_id = ea_id("END", f"{connector_id}:target")
        assoc.set("memberEnd", f"{end1_id} {end2_id}")

        source_end = SubElement(assoc, "ownedEnd")
        source_end.set(qname(NS_XMI, "id"), end1_id)
        source_end.set(qname(NS_XMI, "type"), "uml:Property")
        source_end.set("type", source_id)
        source_end.set("association", connector_id)
        if text(conn.get("lt")):
            source_end.set("name", text(conn.get("lt")))

        target_end = SubElement(assoc, "ownedEnd")
        target_end.set(qname(NS_XMI, "id"), end2_id)
        target_end.set(qname(NS_XMI, "type"), "uml:Property")
        target_end.set("type", target_id)
        target_end.set("association", connector_id)
        if text(conn.get("rt")):
            target_end.set("name", text(conn.get("rt")))

        if relationship_lower == "aggregation":
            target_end.set("aggregation", "shared")
        elif relationship_lower == "composition":
            target_end.set("aggregation", "composite")

        lb_lower, lb_upper = parse_multiplicity(conn.get("lb"))
        rb_lower, rb_upper = parse_multiplicity(conn.get("rb"))

        add_lower_upper(source_end, lb_lower, lb_upper, end1_id)
        add_lower_upper(target_end, rb_lower, rb_upper, end2_id)

        add_comment(assoc, connector_comment_from_conn(conn), f"{connector_id}:comment")

    return tostring(root, encoding="utf-8", xml_declaration=True)


def build_ttl_bytes(model: dict[str, Any], user: str = "", name: str = "") -> bytes:
    ttl_text = _norm_text(model.get("ttl_raw"))
    if ttl_text:
        return ttl_text.encode("utf-8")

    ttl_json = model.get("ttl")
    if ttl_json:
        g = Graph()
        g.parse(data=json.dumps(ttl_json, ensure_ascii=False), format="json-ld")
        ttl_raw = g.serialize(format="turtle")
        return ttl_raw.encode("utf-8") if isinstance(ttl_raw, str) else bytes(ttl_raw)

    g = build_graph_from_xmi_model(
        model,
        user=user,
        name=name or _norm_text(model.get("name")) or "Generated",
    )
    ttl_raw = g.serialize(format="turtle")
    return ttl_raw.encode("utf-8") if isinstance(ttl_raw, str) else bytes(ttl_raw)


def export_model_bytes(user: str = "", name: str = "", target_format: str = "") -> bytes:
    model = get_model(user=user, name=name)
    export_format = (_norm_text(target_format) or _default_source_format(model)).lower()

    if export_format == "xmi":
        return build_xmi_bytes(model)

    if export_format == "ttl":
        return build_ttl_bytes(model, user=user, name=name)

    raise ValueError(f"Unsupported export format: {export_format}")
