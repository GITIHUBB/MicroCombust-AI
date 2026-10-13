import { existsSync, statSync } from "node:fs";
import path from "node:path";

/**
 * The repo lives under `D:\######IMPORTANT`, and Turbopack's built-in CSS
 * resolver mangles `#` in Windows paths by inserting NUL bytes. Tailwind then
 * fails to read its own stylesheet, so the app compiles but ships with zero
 * utility classes and renders completely unstyled.
 *
 * Tailwind consults `globalThis.__tw_resolve` for every stylesheet id *before*
 * falling back to Turbopack's resolver, so installing a plain Node resolver
 * here routes around the bug. Returning `undefined` preserves Tailwind's own
 * resolution as the fallback.
 */
function resolveFrom(specifier, startDir) {
  if (specifier.startsWith(".") || path.isAbsolute(specifier)) {
    const candidate = path.resolve(startDir, specifier);
    return existsSync(candidate) ? candidate : undefined;
  }

  let dir = startDir;
  for (;;) {
    const candidate = path.join(dir, "node_modules", specifier);
    if (existsSync(candidate)) {
      if (!statSync(candidate).isDirectory()) return candidate;
      const indexCss = path.join(candidate, "index.css");
      return existsSync(indexCss) ? indexCss : undefined;
    }
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return undefined;
}

globalThis.__tw_resolve = (specifier, base) => {
  if (typeof specifier !== "string" || !base) return undefined;
  for (const startDir of [base, path.dirname(base)]) {
    const resolved = resolveFrom(specifier, startDir);
    if (resolved) return resolved;
  }
  return undefined;
};

const config = {
  plugins: {
    // Tailwind v4 runs through PostCSS. Configuring it here (instead of the
    // `turbopack.rules` loader that used to live in next.config.ts) keeps the
    // loader inside the project root and lets `build`/`dev` share one config.
    "@tailwindcss/postcss": {},
  },
};

export default config;



