// One shared host environment, independent of project and conversation selection.
$('#environmentMount').innerHTML=`<div id="environmentWizard" class="environment-overlay" hidden><section class="environment-dialog" role="dialog" aria-modal="true" aria-labelledby="envTitle">
 <div class="environment-heading"><div><small data-zh="首次使用 · 全部项目共用" data-en="FIRST RUN · SHARED BY ALL PROJECTS"></small><h2 id="envTitle" data-zh="配置训练环境" data-en="Set up your training environment"></h2></div><div class="environment-heading-actions"><select id="envLanguage" aria-label="Interface language"><option value="zh">中文</option><option value="en">English</option></select><button type="button" id="envLater" class="secondary" data-zh="返回聊天" data-en="Back to chat"></button></div></div>
 <p id="envCurrent"></p><form id="environmentForm">
 <div class="environment-grid"><label><span data-zh="配置方式" data-en="Setup mode"></span><select id="envMode"><option value="create" data-zh="在新文件夹创建环境" data-en="Create an environment in a new folder"></option><option value="existing" data-zh="接入已有环境" data-en="Use an existing environment"></option></select></label><label><span data-zh="训练 GPU 编号" data-en="Training GPU indices"></span><input id="envGpus" value="0" required pattern="[0-9, ]+" aria-label="GPU indices"></label></div>
 <label><span data-zh="Agent 主机上的环境文件夹" data-en="Environment folder on the Agent host"></span><div class="environment-path"><input id="envFolder" required autocomplete="off"><button type="button" id="envBrowse" class="secondary" data-zh="浏览目录" data-en="Browse folders"></button></div></label><p id="envModeHelp" class="environment-help"></p>
 <fieldset><legend data-zh="准备 ImageNet 初始权重" data-en="Prepare ImageNet initial weights"></legend><div class="environment-weights"><label><input name="envWeight" type="checkbox" value="convnext_tiny" checked>ConvNeXt-Tiny</label><label><input name="envWeight" type="checkbox" value="resnet50">ResNet-50</label><label><input name="envWeight" type="checkbox" value="swin_t">Swin-T</label></div><small data-zh="已有缓存会校验后复用，未缓存的所选权重会下载。" data-en="Existing cached files are verified and reused; missing selected weights are downloaded."></small></fieldset>
 <p class="environment-help" data-zh="环境只需配置一次。所有训练入口共用依赖，各项目的数据和训练结果仍各自保存。下载和安装会在后台进行。" data-en="Set up once for all training entries. Each project keeps its own data and results. Downloads and installation run in the background."></p><button id="envStart" type="submit"></button></form>
 <div id="envProgress" hidden><strong id="envPhase" role="status"></strong><p id="envActivity" class="environment-help"></p><progress id="envDownloadBar" max="1" aria-label="ImageNet weight download: current file" hidden></progress><p id="envDownloadText" class="environment-help" hidden></p><p id="envElapsed" class="environment-help"></p><p id="envError" class="error"></p><button type="button" id="envRetryWeights" class="secondary" hidden data-zh="复用此环境，重试权重检查／下载" data-en="Reuse this environment and retry weights"></button><details><summary data-zh="查看配置日志" data-en="View setup log"></summary><pre id="envLog"></pre></details></div>
 <div id="envFolderBrowser" class="environment-browser" hidden><h3 data-zh="浏览 Agent 主机文件夹" data-en="Browse folders on the Agent host"></h3><div class="environment-path"><input id="envBrowsePath" aria-label="Folder path"><button id="envBrowseGo" type="button" class="secondary" data-zh="打开" data-en="Open"></button><button id="envBrowseUp" type="button" class="secondary" data-zh="上一级" data-en="Up"></button></div><div id="envFolderRoots" class="actions"></div><div id="envFolderList"></div><div class="actions"><button id="envBrowseUse" type="button" data-zh="使用此文件夹" data-en="Use this folder"></button><button id="envBrowseClose" type="button" class="secondary" data-zh="取消" data-en="Cancel"></button></div></div>
 </section></div>`;
let environmentFormInitialized=false,environmentSeen=false,environmentDismissed=false,folderListing=null;
function openEnvironment(){environmentDismissed=false;$('#environmentWizard').hidden=false;renderEnvironment();}
function renderEnvironment(){
 const e=state?.environment;if(!e)return;
 $('#envLanguage').value=lang;
 const op=e.operation||{state:'idle'},running=op.state==='running';
 $('#environmentSummary').textContent=e.ready?tr('所有项目共用：','Shared by all projects: ')+e.profile.python:tr('尚未配置训练环境。可先聊天，训练前在这里配置一次。','Training environment is not configured. You can chat first and set it up here before training.');
 $('#envCurrent').textContent=e.ready?tr('当前环境：','Current environment: ')+e.profile.folder:tr('选择一个主机文件夹，所有项目和维护轮共用。','Choose one folder on the Agent host for all projects and maintenance rounds.');
 if(!environmentSeen){environmentSeen=true;if(!e.ready&&!environmentDismissed)$('#environmentWizard').hidden=false;}
 if(!environmentFormInitialized){
  $('#envMode').value=e.ready?'existing':'create';
  $('#envFolder').value=e.profile?.folder||e.default_folder;
  $('#envGpus').value=e.profile?.gpus||'0';
  environmentFormInitialized=true;
 }
 $('#environmentForm').querySelectorAll('input,select,button').forEach(x=>x.disabled=running);
 $('#envStart').textContent=running?tr('正在配置…','Setting up…'):$('#envMode').value==='existing'?tr('检查并使用此环境','Check and use this environment'):tr('创建并配置环境','Create and set up environment');
 $('#envModeHelp').textContent=$('#envMode').value==='existing'?tr('选择已有虚拟环境文件夹，例如 /home/test/ceshi/.venv。只检查，不安装依赖。','Choose an existing venv folder, e.g. /home/test/ceshi/.venv. Dependencies are checked without installation.'):tr('选择空文件夹，或填写要新建的文件夹绝对路径。将在其中创建 .venv，安装共享训练依赖。','Choose an empty folder or enter a new absolute folder path. Its .venv will contain shared training dependencies.');
 const phases={
  'Preparing':['准备环境','Preparing'],
  'Creating the training environment':['创建训练环境','Creating the training environment'],
  'Installing shared training dependencies':['安装共享训练依赖','Installing shared training dependencies'],
  'Checking Python, CUDA/BF16 and requested ImageNet weights':['检查 Python、CUDA/BF16 与所选 ImageNet 权重','Checking Python, CUDA/BF16 and requested ImageNet weights'],
  'Shared environment ready':['共享环境已就绪','Shared environment ready'],
  'Setup failed':['配置失败，请查看下方日志后重试','Setup failed; inspect the log before retrying']};
 $('#envPhase').textContent=phases[op.phase]?tr(...phases[op.phase]):op.phase||'';
 $('#envError').textContent=op.error||'';
 $('#envProgress').hidden=op.state==='idle';
 $('#envLog').textContent=op.log||'';
 const d=op.download,bar=$('#envDownloadBar'),text=$('#envDownloadText');
 bar.hidden=!d;text.hidden=!d;
 if(d){
  const name=({convnext_tiny:'ConvNeXt-Tiny',resnet50:'ResNet-50',swin_t:'Swin-T'})[d.backbone]||d.backbone;
  const label={downloading:tr('下载当前权重文件','Downloading current weight file'),retrying:tr('连接中断，重试下载','Connection interrupted; retrying'),verifying:tr('下载完成，正在校验哈希','Downloaded; verifying hash'),ready:tr('权重校验通过','Weight verified'),cached:tr('已有权重缓存校验通过','Cached weight verified')}[d.status]||d.status;
  const bytes=n=>(n/1048576).toFixed(1)+' MiB';
  text.textContent=`${label} · ${name} · ${bytes(d.received)}${d.total?' / '+bytes(d.total):''}${d.attempt?' · '+tr('尝试','attempt')+' '+d.attempt+'/3':''}`;
  if(d.total)bar.value=Math.max(0,Math.min(1,d.received/d.total));else bar.removeAttribute?.('value');
 }
 $('#envActivity').textContent=!d&&running?(op.activity||tr('配置进程正在运行，详细步骤见日志。','Setup is running; inspect the log for the current step.')):'';
 const seconds=op.elapsed_seconds||0;$('#envElapsed').textContent=op.state==='idle'?'':tr('配置已用时：','Setup elapsed: ')+Math.floor(seconds/60)+'m '+seconds%60+'s';
 $('#envRetryWeights').hidden=!(op.state==='failed'&&d);$('#envRetryWeights').disabled=running;
}
async function environmentRequest(url,options){
 const r=await fetch(url,options),data=await r.json();if(!r.ok)throw Error(data.error||'Environment request failed');return data;
}
$('#environmentForm').onsubmit=e=>{e.preventDefault();return handle(async()=>{
 const payload={folder:$('#envFolder').value.trim(),mode:$('#envMode').value,gpus:$('#envGpus').value.trim(),
  weights:$$('#environmentForm [name="envWeight"]:checked').map(x=>x.value),enable_workers:true};
 state.environment=await environmentRequest('/api/environment/setup',{method:'POST',headers:{'Content-Type':'application/json','X-Facade-CSRF':state.csrf},body:JSON.stringify(payload)});
 renderEnvironment();await refresh();
});};
$('#envMode').onchange=renderEnvironment;
$('#envLanguage').onchange=e=>{$('#language').value=e.target.value;$('#language').onchange(e);};
$('#environmentOpen').onclick=openEnvironment;
$('#envRetryWeights').onclick=()=>{$('#envMode').value='existing';$('#environmentForm').requestSubmit();};
$('#envLater').onclick=()=>{environmentDismissed=true;$('#environmentWizard').hidden=true;};
async function listEnvironmentFolders(path){
 folderListing=await environmentRequest('/api/environment/folders?path='+encodeURIComponent(path||''));
 $('#envFolderBrowser').hidden=false;$('#envBrowsePath').value=folderListing.path;
 $('#envFolderRoots').innerHTML=folderListing.roots.map(p=>`<button type="button" data-env-path="${esc(p)}">${esc(p)}</button>`).join('');
 $('#envFolderList').innerHTML=folderListing.folders.map(f=>`<button type="button" data-env-path="${esc(f.path)}">▱ ${esc(f.name)}</button>`).join('')||`<p>${tr('此目录下没有子文件夹。','No subfolders here.')}</p>`;
}
$('#envBrowse').onclick=()=>handle(()=>listEnvironmentFolders());
$('#envBrowseGo').onclick=()=>handle(()=>listEnvironmentFolders($('#envBrowsePath').value));
$('#envBrowseUp').onclick=()=>handle(()=>listEnvironmentFolders(folderListing.parent));
$('#envBrowseUse').onclick=()=>{$('#envFolder').value=folderListing.path;$('#envFolderBrowser').hidden=true;};
$('#envBrowseClose').onclick=()=>{$('#envFolderBrowser').hidden=true;};
for(const id of ['#envFolderList','#envFolderRoots'])$(id).onclick=e=>{const b=e.target.closest('button[data-env-path]');if(b)handle(()=>listEnvironmentFolders(b.dataset.envPath));};
if(state){translate();renderEnvironment();}
