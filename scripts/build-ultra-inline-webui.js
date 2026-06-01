const fs = require('fs');
const path = require('path');
const vm = require('vm');
const zlib = require('zlib');
const terser = require('terser');

const ROOT = path.resolve(__dirname, '..');
const htmlPath = path.join(ROOT, 'web-ui', 'inscription.html');
const weightsPath = path.join(ROOT, 'pleb.slop');
const outPath = path.join(ROOT, 'web-ui', 'inscription.ultra.html');

const sourceHtml = fs.readFileSync(htmlPath, 'utf8');
const sourceScript = sourceHtml.match(/<script>([\s\S]*?)<\/script>/)[1];
const sourceVocab = vm.runInNewContext(
  sourceScript.match(/const VOCAB = \[([\s\S]*?)\];/)[0] + ';VOCAB'
);

function section(start, end) {
  const a = sourceScript.indexOf(start);
  const b = sourceScript.indexOf(end, a);
  if (a === -1 || b === -1) throw new Error(`Could not extract section: ${start}`);
  return sourceScript.slice(a, b);
}

const tokenizer = section('    const CONTRACTIONS = ', '    // ============================================================================\n    // MODEL CONFIGURATION');
const modelRuntime = section('    const GENERATION_VOCAB_SIZE = ', '    // ============================================================================\n    // APPLICATION');

const appRuntime = `
let VOCAB=[],STOI=new Map(),ITOS=new Map(),UNK_ID=0;
function initVocab(v){VOCAB=v;STOI=new Map();ITOS=new Map();VOCAB.forEach((w,i)=>{STOI.set(w,i);ITOS.set(i,w)});UNK_ID=STOI.get('<UNK>')||0}
${tokenizer}
${modelRuntime}
const $=id=>document.getElementById(id),out=$('o'),btn=$('g'),promptSelect=$('p'),custom=$('c'),temp=$('t'),topP=$('q'),topK=$('k'),freq=$('f');
function num(el,fb,min,max,integer){let v=Number(el.value);if(!Number.isFinite(v))v=fb;v=Math.min(Math.max(v,min),max);return integer?Math.round(v):v}
const marker=new TextEncoder().encode('<!--SLOP_PAYLOAD_V1-->');
function lastIndexOfBytes(haystack,needle){for(let i=haystack.length-needle.length;i>=0;i--){let ok=true;for(let j=0;j<needle.length;j++){if(haystack[i+j]!==needle[j]){ok=false;break}}if(ok)return i}return-1}
async function inflateText(bytes){const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'));return new TextDecoder().decode(await new Response(stream).arrayBuffer())}
async function loadTail(){const response=await fetch(location.href);const bytes=new Uint8Array(await response.arrayBuffer());const markerAt=lastIndexOfBytes(bytes,marker);if(markerAt<0)throw new Error('missing embedded payload');const headerAt=markerAt+marker.length;const view=new DataView(bytes.buffer,bytes.byteOffset+headerAt,8);const modelLen=view.getUint32(0,true),vocabLen=view.getUint32(4,true);const dataAt=headerAt+8;return{modelBytes:bytes.slice(dataAt,dataAt+modelLen).buffer,vocabBytes:bytes.slice(dataAt+modelLen,dataAt+modelLen+vocabLen)}}
async function init(){try{out.textContent='Loading model...';const payload=await loadTail();initVocab((await inflateText(payload.vocabBytes)).split('\\n'));model=new SlopModel();model._loadTensors(await SlopParser.parse(payload.modelBytes));model.loaded=true;out.textContent='Ready.';btn.disabled=false}catch(e){out.textContent='Error: '+e.message;console.error(e)}}
let model=null;
promptSelect.onchange=()=>{custom.hidden=promptSelect.value!=='custom'};
btn.onclick=async()=>{const started=performance.now();btn.disabled=true;out.textContent='Generating...';try{const prompt=promptSelect.value==='custom'?custom.value.trim():promptSelect.value;const ids=encode(prompt||'bitcoin is');const result=await model.generate(ids,32,num(temp,.8,.1,2),num(topK,100,1,2000,true),num(topP,.9,.05,1),num(freq,.5,0,2));out.textContent=decode(result)+'\\n\\n(Time: '+Math.round(performance.now()-started)+'ms)'}catch(e){out.textContent='Error: '+e.message}finally{btn.disabled=false}};
init();
`;

async function main() {
  const minified = await terser.minify(appRuntime, {
    ecma: 2020,
    compress: {
      passes: 3,
      unsafe: true,
      unsafe_arrows: true,
    },
    mangle: true,
    format: { comments: false },
  });
  if (minified.error) throw minified.error;

  const html = `<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Pleb Slop</title><style>html,body{background:#fff;color:#111}body{font:16px system-ui;margin:20px;max-width:760px}select,input,button{font:inherit;margin:4px 0;padding:8px}button{display:block}pre{white-space:pre-wrap;border:1px solid #ddd;padding:12px;min-height:12em}</style><h1>Onchain Plebslop Generator</h1><label>prompt <select id=p><option>bitcoin is</option><option>the problem with fiat</option><option>in the future</option><option>satoshi</option><option>hyperbitcoinization means</option><option value=custom>custom</option></select></label><input id=c hidden><button id=g disabled>Generate</button><pre id=o>Loading...</pre><h2>Advanced sampler controls</h2><label>temperature <input id=t type=number min=.1 max=2 step=.05 value=.8></label><label>top-p <input id=q type=number min=.05 max=1 step=.05 value=.9></label><label>top-k <input id=k type=number min=1 max=2000 step=1 value=100></label><label>frequency penalty <input id=f type=number min=0 max=2 step=.05 value=.5></label><h2>What is this?</h2><p>Generate pleb slop with model weights stored in a single standard bitcoin transaction.</p><p>A tiny transformer for manufacturing Bitcoin brainworms locally. Pick a prompt, hit Generate, and the page coughs up compact maxi rhetoric without phoning a server.</p><p>The training diet is exactly what you fear: monetary grievance, proof-of-work sermons, sovereignty cosplay, custody paranoia, and other low-grade signal distilled into token soup.</p><p>The trick is the artifact. UI, tokenizer, inference runtime, compressed vocabulary, and quantized weights all ride in one file under the standard transaction limit. The model is not remote. The slop is self-contained.</p><pre>Vocabulary Size: 2000 generation tokens
Embedding Dimension: 112
Transformer Layers: 4
Attention Heads: 4
Head Dimension: 28
Context Length: 128 tokens
FFN Hidden Dimension: 448
Generation Length: 32 tokens
Quantization: 4-bit signed nibbles</pre><script>${minified.code}</script>`;

  const inflatedWeights = zlib.inflateSync(fs.readFileSync(weightsPath));
  const optimizedWeights = zlib.deflateSync(inflatedWeights, {
    level: 9,
    strategy: zlib.constants.Z_FILTERED,
  });
  const vocabBytes = zlib.deflateSync(sourceVocab.join('\n'), { level: 9 });
  const marker = Buffer.from('<template hidden><!--SLOP_PAYLOAD_V1-->');
  const header = Buffer.alloc(8);
  header.writeUInt32LE(optimizedWeights.length, 0);
  header.writeUInt32LE(vocabBytes.length, 4);

  const output = Buffer.concat([
    Buffer.from(html),
    marker,
    header,
    optimizedWeights,
    vocabBytes,
  ]);
  fs.writeFileSync(outPath, output);

  console.log(`runtime:  ${Buffer.byteLength(minified.code).toLocaleString()} bytes`);
  console.log(`html:     ${Buffer.byteLength(html).toLocaleString()} bytes`);
  console.log(`weights:  ${optimizedWeights.length.toLocaleString()} bytes`);
  console.log(`vocab:    ${vocabBytes.length.toLocaleString()} bytes`);
  console.log(`output:   ${output.length.toLocaleString()} bytes`);
  console.log(`slack400: ${(400000 - output.length).toLocaleString()} bytes`);
  console.log(`slack409: ${(409600 - output.length).toLocaleString()} bytes`);
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
