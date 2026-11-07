#!/usr/bin/env bash
# Build docs/solution.pdf from docs/solution.md (needs pandoc + playwright chromium)
set -euo pipefail
cd "$(dirname "$0")"
pandoc solution.md -s --metadata pagetitle="Promo Finder" -c style.css --embed-resources -o solution.html
node -e "
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch(); const p = await b.newPage();
  await p.goto('file://' + process.cwd() + '/solution.html');
  await p.pdf({ path: 'solution.pdf', format: 'A4', preferCSSPageSize: true });
  await b.close();
})();"
pdfinfo solution.pdf 2>/dev/null | grep Pages || true
