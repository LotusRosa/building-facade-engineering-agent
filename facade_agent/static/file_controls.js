/* Browser-native file captions follow the OS locale, not the Agent language. */
Object.assign(copy.zh,{file_choose:"选择文件",file_choose_folder:"选择文件夹",file_none:"未选择文件",file_count:"已选择 {count} 个文件",file_required:"请先选择文件。"});
Object.assign(copy.en,{file_choose:"Choose files",file_choose_folder:"Choose folder",file_none:"No files selected",file_count:"{count} files selected",file_required:"Select a file first."});
const localizedFileControls=new WeakMap();
function refreshFileControls() {
  document.querySelectorAll('input[type="file"]').forEach(input=>{
    let controls=localizedFileControls.get(input);
    if(!controls) {
      const wrapper=document.createElement("div");wrapper.className="localized-file-control";
      const button=document.createElement("button");button.type="button";button.className="ghost-button";
      const status=document.createElement("span");status.className="localized-file-status";status.setAttribute("aria-live","polite");
      input.classList.add("localized-file-native");input.tabIndex=-1;
      input.parentNode.insertBefore(wrapper,input);wrapper.append(input,button,status);
      button.onclick=event=>{event.preventDefault();if(!input.disabled)input.click()};
      input.addEventListener("change",refreshFileControls);
      input.addEventListener("invalid",event=>{event.preventDefault();button.focus();toast(t("file_required"))});
      controls={button,status};localizedFileControls.set(input,controls);
    }
    const title=t(input.hasAttribute("webkitdirectory")?"file_choose_folder":"file_choose");
    const files=[...input.files],summary=files.length===1?files[0].name:files.length?t("file_count").replace("{count}",String(files.length)):t("file_none");
    if(controls.button.textContent!==title)controls.button.textContent=title;
    if(controls.status.textContent!==summary)controls.status.textContent=summary;
    if(controls.button.disabled!==input.disabled)controls.button.disabled=input.disabled;
    const field=input.closest(".field")?.querySelector("span");
    controls.button.setAttribute("aria-label",`${title}${field?`: ${field.textContent}`:""}`);
  });
}
let fileControlsRefreshPending=false;
new MutationObserver(()=>{
  if(fileControlsRefreshPending)return;
  fileControlsRefreshPending=true;
  requestAnimationFrame(()=>{fileControlsRefreshPending=false;refreshFileControls()});
}).observe(document.body,{childList:true,subtree:true,attributes:true,attributeFilter:["disabled"]});
refreshFileControls();
