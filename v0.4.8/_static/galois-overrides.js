(function () {
    // Immediately add a class to hide property labels before first paint
    document.documentElement.classList.add("galois-hide-property-labels");

    function should_make_class_property(dt) {
        // Check the "galois.FieldArray." or "galois.Array." prefix in the prename
        const prename = dt.querySelector(".sig-prename.descclassname");
        if (prename) {
            const txt = prename.textContent || "";
            if (/galois\.(FieldArray|Array)\./.test(txt)) {
                return true;
            }
        }

        // Fallback: check href on the name link (summary pages)
        const link = dt.querySelector("a.sig-name, a.sig-name.descname, a.sig-name.reference");
        if (link) {
            const href = link.getAttribute("href") || "";
            if (/galois\.FieldArray\./.test(href) || /galois\.Array\./.test(href)) {
                return true;
            }
        }

        return false;
    }

    function rewrite_labels() {
        // Look at both summary listings and detail pages
        const dls = document.querySelectorAll(
            "dl.py.property.summary.objdesc, dl.py.property.objdesc"
        );

        dls.forEach((dl) => {
            const dt = dl.querySelector("dt");
            if (!dt) return;
            if (!should_make_class_property(dt)) return;

            // Find the inner "property" label span
            const labelNode =
                dt.querySelector("em.property .k .pre") ||
                dt.querySelector("em.property .k");

            if (!labelNode) return;

            const text = (labelNode.textContent || "").trim();
            if (text === "property") {
                labelNode.textContent = "class property";
            }
        });

        // Reveal labels now that text is correct
        document.documentElement.classList.remove("galois-hide-property-labels");
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", rewrite_labels);
    } else {
        rewrite_labels();
    }
})();
