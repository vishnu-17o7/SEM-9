const fs = require("fs");
const path = require("path");
const assert = require("assert");

const root = path.resolve(__dirname, "..");
const read = (file) => fs.readFileSync(path.join(root, file), "utf8");

const manifest = JSON.parse(read("manifest.json"));

assert.strictEqual(manifest.manifest_version, 3);
assert.strictEqual(manifest.background.service_worker, "background.js");
assert.strictEqual(manifest.action.default_popup, "popup.html");
assert(manifest.permissions.includes("activeTab"));
assert(manifest.permissions.includes("tabs"));
assert(manifest.host_permissions.includes("http://127.0.0.1:5000/*"));

for (const file of ["background.js", "popup.html", "popup.css", "popup.js", "README.md"]) {
  assert(fs.existsSync(path.join(root, file)), `${file} is missing`);
}

const popupHtml = read("popup.html");
assert(popupHtml.includes('<script src="popup.js"></script>'));
assert(!popupHtml.includes("onclick="), "popup.html should not use inline event handlers");

const backgroundJs = read("background.js");
assert(backgroundJs.includes("http://127.0.0.1:5000/predict"));
assert(backgroundJs.includes("chrome.tabs.onUpdated"));
assert(backgroundJs.includes("chrome.runtime.onMessage"));

console.log("Extension contract checks passed.");
