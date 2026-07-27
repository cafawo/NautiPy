(function () {
  "use strict";

  var SVG_NAMESPACE = "http://www.w3.org/2000/svg";
  var WIDTH = 720;
  var HEIGHT = 430;
  var PLOT = { left: 48, top: 28, width: 624, height: 328 };

  function svgElement(name, attributes, text) {
    var element = document.createElementNS(SVG_NAMESPACE, name);
    Object.keys(attributes || {}).forEach(function (key) {
      element.setAttribute(key, String(attributes[key]));
    });
    if (typeof text === "string") {
      element.textContent = text;
    }
    return element;
  }

  function htmlElement(name, className, text) {
    var element = document.createElement(name);
    if (className) {
      element.className = className;
    }
    if (typeof text === "string") {
      element.textContent = text;
    }
    return element;
  }

  function finiteNumber(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  function validateDocument(documentData) {
    if (
      !documentData ||
      documentData.schema_version !== 1 ||
      !Array.isArray(documentData.scenarios) ||
      documentData.scenarios.length === 0
    ) {
      throw new Error("Unsupported or empty Fix Lab fixture");
    }
    documentData.scenarios.forEach(function (scenario) {
      if (
        !scenario ||
        typeof scenario.id !== "string" ||
        typeof scenario.title !== "string" ||
        !Array.isArray(scenario.references) ||
        !Array.isArray(scenario.observations) ||
        !Array.isArray(scenario.positions)
      ) {
        throw new Error("Malformed Fix Lab scenario");
      }
    });
    return documentData;
  }

  function allPoints(scenario) {
    return scenario.references.concat(scenario.positions);
  }

  function pointById(scenario, identifier) {
    return allPoints(scenario).find(function (point) {
      return point.id === identifier;
    });
  }

  function boundsFor(scenario) {
    var points = allPoints(scenario);
    var minX = Infinity;
    var maxX = -Infinity;
    var minY = Infinity;
    var maxY = -Infinity;

    function include(x, y, radiusX, radiusY) {
      if (!finiteNumber(x) || !finiteNumber(y)) {
        return;
      }
      radiusX = finiteNumber(radiusX) ? Math.abs(radiusX) : 0;
      radiusY = finiteNumber(radiusY) ? Math.abs(radiusY) : radiusX;
      minX = Math.min(minX, x - radiusX);
      maxX = Math.max(maxX, x + radiusX);
      minY = Math.min(minY, y - radiusY);
      maxY = Math.max(maxY, y + radiusY);
    }

    points.forEach(function (point) {
      include(point.east_m, point.north_m);
    });
    scenario.observations.forEach(function (observation) {
      if (observation.kind !== "range") {
        return;
      }
      var reference = pointById(scenario, observation.reference_id);
      if (reference) {
        include(
          reference.east_m,
          reference.north_m,
          observation.value,
          observation.value
        );
      }
    });
    (scenario.ellipses || []).forEach(function (ellipse) {
      var center = pointById(scenario, ellipse.center_id);
      if (center) {
        var radius = ellipse.semi_major_95_m * (ellipse.display_scale || 1);
        include(center.east_m, center.north_m, radius, radius);
      }
    });

    if (![minX, maxX, minY, maxY].every(finiteNumber)) {
      return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
    }
    var spanX = Math.max(maxX - minX, 1);
    var spanY = Math.max(maxY - minY, 1);
    var margin = Math.max(spanX, spanY) * 0.08;
    return {
      minX: minX - margin,
      maxX: maxX + margin,
      minY: minY - margin,
      maxY: maxY + margin,
    };
  }

  function projectionFor(scenario) {
    var bounds = boundsFor(scenario);
    var scale = Math.min(
      PLOT.width / (bounds.maxX - bounds.minX),
      PLOT.height / (bounds.maxY - bounds.minY)
    );
    var centerX = (bounds.minX + bounds.maxX) / 2;
    var centerY = (bounds.minY + bounds.maxY) / 2;

    return {
      scale: scale,
      point: function (east, north) {
        return {
          x: PLOT.left + PLOT.width / 2 + (east - centerX) * scale,
          y: PLOT.top + PLOT.height / 2 - (north - centerY) * scale,
        };
      },
    };
  }

  function addDefinitions(svg, instanceId) {
    var definitions = svgElement("defs");
    var arrow = svgElement("marker", {
      id: instanceId + "-bearing-arrow",
      viewBox: "0 0 10 10",
      refX: "9",
      refY: "5",
      markerWidth: "7",
      markerHeight: "7",
      orient: "auto-start-reverse",
    });
    arrow.appendChild(
      svgElement("path", {
        d: "M 0 0 L 10 5 L 0 10 z",
        class: "fix-lab__arrow-head",
      })
    );
    definitions.appendChild(arrow);
    svg.appendChild(definitions);
  }

  function drawAxes(svg) {
    var group = svgElement("g", {
      class: "fix-lab__axes",
      "aria-hidden": "true",
    });
    group.appendChild(
      svgElement("path", {
        d: "M 55 382 H 117 M 55 382 V 320",
      })
    );
    group.appendChild(svgElement("text", { x: 123, y: 387 }, "E"));
    group.appendChild(svgElement("text", { x: 48, y: 313 }, "N"));
    group.appendChild(
      svgElement(
        "text",
        { x: 150, y: 388, class: "fix-lab__axis-note" },
        "local schematic · not a chart"
      )
    );
    svg.appendChild(group);
  }

  function drawRanges(svg, scenario, projection) {
    var group = svgElement("g", {
      class: "fix-lab__ranges",
      "aria-hidden": "true",
    });
    scenario.observations.forEach(function (observation, index) {
      if (observation.kind !== "range") {
        return;
      }
      var reference = pointById(scenario, observation.reference_id);
      if (!reference) {
        return;
      }
      var center = projection.point(reference.east_m, reference.north_m);
      group.appendChild(
        svgElement("circle", {
          cx: center.x,
          cy: center.y,
          r: observation.value * projection.scale,
          class:
            "fix-lab__range-circle fix-lab__range-circle--" +
            String((index % 3) + 1),
        })
      );
    });
    svg.appendChild(group);
  }

  function drawBearings(svg, scenario, projection, instanceId) {
    var group = svgElement("g", {
      class: "fix-lab__bearings",
      "aria-hidden": "true",
    });
    scenario.observations.forEach(function (observation) {
      if (observation.kind !== "bearing") {
        return;
      }
      var origin = pointById(scenario, observation.origin_id);
      var reference = pointById(scenario, observation.reference_id);
      if (!origin || !reference) {
        return;
      }
      var start = projection.point(origin.east_m, origin.north_m);
      var end = projection.point(reference.east_m, reference.north_m);
      group.appendChild(
        svgElement("line", {
          x1: start.x,
          y1: start.y,
          x2: end.x,
          y2: end.y,
          class: "fix-lab__bearing-line",
          "data-bearing-origin": observation.origin_id,
          "marker-end": "url(#" + instanceId + "-bearing-arrow)",
        })
      );
    });
    svg.appendChild(group);
  }

  function drawEllipses(svg, scenario, projection) {
    var group = svgElement("g", {
      class: "fix-lab__ellipses",
      "aria-hidden": "true",
    });
    (scenario.ellipses || []).forEach(function (ellipse, index) {
      var centerPoint = pointById(scenario, ellipse.center_id);
      if (!centerPoint) {
        return;
      }
      var center = projection.point(centerPoint.east_m, centerPoint.north_m);
      var displayScale = ellipse.display_scale || 1;
      var bearing = finiteNumber(ellipse.major_axis_bearing)
        ? ellipse.major_axis_bearing
        : 90;
      group.appendChild(
        svgElement("ellipse", {
          cx: center.x,
          cy: center.y,
          rx: Math.max(
            ellipse.semi_major_95_m * displayScale * projection.scale,
            1.5
          ),
          ry: Math.max(
            ellipse.semi_minor_95_m * displayScale * projection.scale,
            1.5
          ),
          transform:
            "rotate(" +
            String(bearing - 90) +
            " " +
            String(center.x) +
            " " +
            String(center.y) +
            ")",
          class:
            "fix-lab__ellipse fix-lab__ellipse--" + String((index % 2) + 1),
        })
      );
    });
    svg.appendChild(group);
  }

  function drawReferences(svg, scenario, projection) {
    var group = svgElement("g", { class: "fix-lab__references" });
    scenario.references.forEach(function (reference) {
      var position = projection.point(reference.east_m, reference.north_m);
      var marker = svgElement("g", {
        class: "fix-lab__reference",
        transform: "translate(" + position.x + " " + position.y + ")",
      });
      marker.appendChild(
        svgElement("path", {
          d: "M -7 7 L 0 -8 L 7 7 Z M -10 9 H 10",
          "aria-hidden": "true",
        })
      );
      marker.appendChild(
        svgElement(
          "text",
          { x: 11, y: -9, class: "fix-lab__marker-label" },
          reference.label
        )
      );
      group.appendChild(marker);
    });
    svg.appendChild(group);
  }

  function positionShape(group, point) {
    if (point.kind === "truth") {
      group.appendChild(
        svgElement("path", {
          d: "M -9 7 L 0 -9 L 9 7 Z M -12 10 Q 0 15 12 10",
          "aria-hidden": "true",
        })
      );
    } else if (point.kind === "fix") {
      group.appendChild(
        svgElement("path", {
          d: "M 0 -9 L 9 0 L 0 9 L -9 0 Z",
          "aria-hidden": "true",
        })
      );
    } else if (point.kind === "comparison") {
      group.appendChild(
        svgElement("rect", {
          x: -8,
          y: -8,
          width: 16,
          height: 16,
          "aria-hidden": "true",
        })
      );
    } else {
      group.appendChild(
        svgElement("circle", {
          cx: 0,
          cy: 0,
          r: 8,
          "aria-hidden": "true",
        })
      );
    }
  }

  function drawPositions(svg, scenario, projection) {
    var group = svgElement("g", { class: "fix-lab__positions" });
    scenario.positions.forEach(function (point, index) {
      var position = projection.point(point.east_m, point.north_m);
      var marker = svgElement("g", {
        class: "fix-lab__position fix-lab__position--" + point.kind,
        transform: "translate(" + position.x + " " + position.y + ")",
      });
      var accessibleTitle = svgElement("title", {}, point.label);
      marker.appendChild(accessibleTitle);
      positionShape(marker, point);
      marker.appendChild(
        svgElement(
          "text",
          {
            x: 12,
            y: 18 + (index % 2) * 14,
            class: "fix-lab__marker-label",
          },
          point.label
        )
      );
      group.appendChild(marker);
    });
    svg.appendChild(group);
  }

  function renderVisual(container, scenario, instanceId) {
    var projection = projectionFor(scenario);
    var svg = svgElement("svg", {
      class: "fix-lab__plot",
      viewBox: "0 0 " + WIDTH + " " + HEIGHT,
      role: "img",
      "aria-labelledby": instanceId + "-title " + instanceId + "-description",
    });
    svg.appendChild(
      svgElement(
        "title",
        { id: instanceId + "-title" },
        scenario.title
      )
    );
    svg.appendChild(
      svgElement(
        "desc",
        { id: instanceId + "-description" },
        scenario.lesson + " " + scenario.schematic_note
      )
    );
    addDefinitions(svg, instanceId);
    drawRanges(svg, scenario, projection);
    drawBearings(svg, scenario, projection, instanceId);
    drawEllipses(svg, scenario, projection);
    drawReferences(svg, scenario, projection);
    drawPositions(svg, scenario, projection);
    drawAxes(svg);

    container.replaceChildren(svg);
    if (container.scrollWidth > container.clientWidth) {
      container.scrollLeft = (container.scrollWidth - container.clientWidth) / 2;
    }
  }

  function formatted(value, digits) {
    if (!finiteNumber(value)) {
      return "not available";
    }
    return value.toLocaleString(undefined, {
      maximumFractionDigits: digits,
    });
  }

  function addDiagnostic(list, term, value) {
    var wrapper = htmlElement("div", "fix-lab__diagnostic");
    wrapper.appendChild(htmlElement("dt", "", term));
    wrapper.appendChild(htmlElement("dd", "", value));
    list.appendChild(wrapper);
  }

  function renderResidualTable(residuals) {
    if (!Array.isArray(residuals) || residuals.length === 0) {
      return null;
    }
    var wrapper = htmlElement("div", "fix-lab__table-wrap");
    var table = htmlElement("table", "fix-lab__residuals");
    var caption = htmlElement(
      "caption",
      "",
      "Observation residuals: prediction minus observation"
    );
    var head = htmlElement("thead");
    var headRow = htmlElement("tr");
    ["Observation", "Natural residual", "Standardized residual"].forEach(
      function (label) {
        var header = htmlElement("th", "", label);
        header.setAttribute("scope", "col");
        headRow.appendChild(header);
      }
    );
    head.appendChild(headRow);
    table.appendChild(caption);
    table.appendChild(head);

    var body = htmlElement("tbody");
    residuals.forEach(function (residual) {
      var row = htmlElement("tr");
      var label = htmlElement("th", "", String(residual.label || "Observation"));
      label.setAttribute("scope", "row");
      row.appendChild(label);
      row.appendChild(
        htmlElement(
          "td",
          "",
          formatted(
            residual.natural_residual,
            residual.kind === "bearing" ? 4 : 2
          ) +
            " " +
            String(residual.natural_unit || "")
        )
      );
      row.appendChild(
        htmlElement(
          "td",
          "",
          formatted(residual.standardized_residual, 4) + " σ"
        )
      );
      body.appendChild(row);
    });
    table.appendChild(body);
    wrapper.appendChild(table);
    return wrapper;
  }

  function renderSummary(container, scenario) {
    var diagnostics = scenario.diagnostics || {};
    var heading = htmlElement("h3", "fix-lab__scenario-title", scenario.title);
    var lesson = htmlElement("p", "fix-lab__lesson", scenario.lesson);
    var note = htmlElement("p", "fix-lab__schematic-note", scenario.schematic_note);
    var list = htmlElement("dl", "fix-lab__diagnostics");

    addDiagnostic(list, "Outcome", String(diagnostics.status || "not available"));
    addDiagnostic(
      list,
      "Observations",
      String(scenario.observations.length)
    );
    if (finiteNumber(diagnostics.condition_number)) {
      addDiagnostic(
        list,
        "Condition number",
        formatted(diagnostics.condition_number, 3)
      );
    }
    if (finiteNumber(diagnostics.rms)) {
      addDiagnostic(
        list,
        "Standardized RMS",
        formatted(diagnostics.rms, 4)
      );
    }
    if (finiteNumber(diagnostics.candidate_count)) {
      addDiagnostic(
        list,
        "Candidates",
        String(diagnostics.candidate_count)
      );
    }
    if (finiteNumber(diagnostics.high_weight_shift_m)) {
      addDiagnostic(
        list,
        "Shift with σ = 1 m",
        formatted(diagnostics.high_weight_shift_m, 1) + " m"
      );
      addDiagnostic(
        list,
        "Shift with σ = 1,000 m",
        formatted(diagnostics.low_weight_shift_m, 3) + " m"
      );
    }
    if (finiteNumber(diagnostics.scaled_semi_major_95_m)) {
      addDiagnostic(
        list,
        "95% semi-major axis",
        formatted(diagnostics.baseline_semi_major_95_m, 1) +
          " m → " +
          formatted(diagnostics.scaled_semi_major_95_m, 1) +
          " m"
      );
    }

    var fragments = [heading, lesson, list];
    var residualTable = renderResidualTable(diagnostics.residuals);
    if (residualTable) {
      fragments.push(residualTable);
    }
    if (Array.isArray(diagnostics.warnings) && diagnostics.warnings.length) {
      var warning = htmlElement(
        "p",
        "fix-lab__warning",
        "Diagnostic: " + diagnostics.warnings.join("; ")
      );
      fragments.push(warning);
    }
    fragments.push(note);
    container.replaceChildren.apply(container, fragments);
  }

  function createControls(container, scenarios, instanceId, onChange) {
    var label = htmlElement("label", "fix-lab__label", "Choose a scenario");
    var select = htmlElement("select", "fix-lab__select");
    select.id = instanceId + "-scenario";
    select.setAttribute("data-fix-lab-select", "");
    label.setAttribute("for", select.id);
    scenarios.forEach(function (scenario) {
      var option = htmlElement("option", "", scenario.title);
      option.value = scenario.id;
      select.appendChild(option);
    });
    select.addEventListener("change", function () {
      onChange(select.value);
    });
    container.replaceChildren(label, select);
    container.hidden = false;
    return select;
  }

  function initialize(root, rootIndex) {
    var fixtureUrl = root.getAttribute("data-fixture-url");
    var status = root.querySelector("[data-fix-lab-status]");
    var controls = root.querySelector("[data-fix-lab-controls]");
    var visual = root.querySelector("[data-fix-lab-visual]");
    var summary = root.querySelector("[data-fix-lab-summary]");
    var fallback = root.querySelector("[data-fix-lab-fallback]");
    var instanceId = "fix-lab-" + String(rootIndex + 1);

    if (!fixtureUrl || !status || !controls || !visual || !summary) {
      return;
    }
    status.textContent = "Loading the interactive scenarios…";
    controls.hidden = true;

    fetch(new URL(fixtureUrl, document.baseURI), {
      headers: { Accept: "application/json" },
    })
      .then(function (response) {
        if (!response.ok) {
          throw new Error("Fixture request failed with " + response.status);
        }
        return response.json();
      })
      .then(validateDocument)
      .then(function (documentData) {
        var scenarios = documentData.scenarios;
        var byId = new Map(
          scenarios.map(function (scenario) {
            return [scenario.id, scenario];
          })
        );
        var render = function (identifier) {
          var scenario = byId.get(identifier) || scenarios[0];
          renderVisual(visual, scenario, instanceId);
          renderSummary(summary, scenario);
          status.textContent =
            "Showing “" +
            scenario.title +
            "”. Results were precomputed on WGS84.";
        };
        var select = createControls(
          controls,
          scenarios,
          instanceId,
          render
        );
        if (fallback) {
          fallback.hidden = true;
        }
        select.value = scenarios[0].id;
        render(scenarios[0].id);
      })
      .catch(function () {
        status.textContent =
          "The interactive comparison is unavailable; the complete static " +
          "explanation remains below.";
        controls.hidden = true;
        visual.replaceChildren();
        summary.replaceChildren();
        if (fallback) {
          fallback.hidden = false;
        }
      });
  }

  document.querySelectorAll("[data-fix-lab]").forEach(initialize);
})();
