const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const ROOT = path.resolve(__dirname, '..');
const OUT_DIR = path.join(ROOT, 'dist', 'inscription');

const files = [
  {
    role: 'parent-ui',
    source: path.join(ROOT, 'web-ui', 'inscription.html'),
    output: 'pleb-slop-ui.html',
    expectedContentType: 'text/html;charset=utf-8',
  },
  {
    role: 'child-weights',
    source: path.join(ROOT, 'pleb.slop'),
    output: 'pleb-slop-weights.bin',
    expectedContentType: 'application/octet-stream',
  },
];

function sha256(buffer) {
  return crypto.createHash('sha256').update(buffer).digest('hex');
}

fs.rmSync(OUT_DIR, { recursive: true, force: true });
fs.mkdirSync(OUT_DIR, { recursive: true });

const manifest = {
  generatedAt: new Date().toISOString(),
  instructions: [
    'Inscribe pleb-slop-ui.html first as the parent.',
    'After the parent inscription is confirmed and you have its inscription ID, inscribe pleb-slop-weights.bin as a child of that parent.',
    'The child must be served as application/octet-stream. If your web tool lets you override content type, set it explicitly.',
  ],
  files: [],
};

for (const file of files) {
  const bytes = fs.readFileSync(file.source);
  const outputPath = path.join(OUT_DIR, file.output);
  fs.writeFileSync(outputPath, bytes);
  manifest.files.push({
    role: file.role,
    file: file.output,
    source: path.relative(ROOT, file.source),
    bytes: bytes.length,
    sha256: sha256(bytes),
    expectedContentType: file.expectedContentType,
  });
}

fs.writeFileSync(
  path.join(OUT_DIR, 'manifest.json'),
  `${JSON.stringify(manifest, null, 2)}\n`
);

fs.writeFileSync(
  path.join(OUT_DIR, 'README.txt'),
  `${manifest.instructions.join('\n')}\n\nFiles:\n${manifest.files
    .map(
      file =>
        `- ${file.file}: ${file.role}, ${file.bytes} bytes, sha256 ${file.sha256}, expected ${file.expectedContentType}`
    )
    .join('\n')}\n`
);

for (const file of manifest.files) {
  console.log(
    `${file.file}\t${file.bytes} bytes\t${file.sha256}\t${file.expectedContentType}`
  );
}
console.log(`\nWrote ${path.relative(ROOT, OUT_DIR)}`);
