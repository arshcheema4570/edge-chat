#!/usr/bin/env node
// Build an unsigned Web Bundle from tracked site files. This intentionally does
// not sign the bundle or create/handle any IWA signing keys.
import { execFileSync, spawnSync } from "node:child_process";
import { copyFileSync, lstatSync, mkdirSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import process from "node:process";

const repoRoot = process.cwd();
const repoName = path.basename(repoRoot);
const outputFlag = process.argv.indexOf("--output");
const outputPath = path.resolve(
  outputFlag >= 0 && process.argv[outputFlag + 1]
    ? process.argv[outputFlag + 1]
    : path.join(tmpdir(), "iwa-unsigned", `${repoName}.wbn`),
);
const stagingRoot = mkdtempSync(path.join(tmpdir(), `iwa-${repoName}-`));
const omitted = (name) =>
  name === "README.md" ||
  name === "IWA.md" ||
  name === ".gitignore" ||
  name === ".gitattributes" ||
  name === ".editorconfig" ||
  name === "package.json" ||
  name === "package-lock.json" ||
  name.startsWith(".github/") ||
  name.startsWith("scripts/") ||
  name.startsWith("node_modules/") ||
  name.startsWith("build/") ||
  /^LICENSE(?:\.|$)/i.test(path.posix.basename(name));

try {
  let appName = repoName;
  try {
    const manifest = JSON.parse(readFileSync(path.join(repoRoot, "manifest.json"), "utf8"));
    appName = manifest.name || appName;
  } catch {
    // Some static sites may not have a web app manifest; the repo name is enough.
  }

  const tracked = execFileSync("git", ["ls-files", "-z"], { cwd: repoRoot })
    .toString("utf8")
    .split("\0")
    .filter(Boolean)
    .filter((name) => !omitted(name));
  if (tracked.length === 0) throw new Error("No tracked site files were found.");

  let copied = 0;
  for (const relativeName of tracked) {
    const source = path.join(repoRoot, relativeName);
    const stat = lstatSync(source);
    if (!stat.isFile()) continue;
    const destination = path.join(stagingRoot, relativeName);
    mkdirSync(path.dirname(destination), { recursive: true });
    copyFileSync(source, destination);
    copied++;
  }
  if (!copied) throw new Error("No regular site files were copied into the bundle staging folder.");

  mkdirSync(path.dirname(outputPath), { recursive: true });
  const result = spawnSync(
    "npx",
    ["--yes", "wbn@0.0.9", "--dir", stagingRoot, "--output", outputPath],
    { cwd: repoRoot, stdio: "inherit" },
  );
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`wbn exited with status ${result.status}`);

  console.log(`Unsigned bundle created for ${appName}`);
  console.log(`Included ${copied} tracked site files.`);
  console.log(`Output: ${outputPath}`);
  console.log("This .wbn is unsigned and cannot be installed as an IWA.");
} finally {
  rmSync(stagingRoot, { recursive: true, force: true });
}
