# 连接 LLM

右上角“连接模型”可选择 DeepSeek、OpenAI 或 OpenAI 兼容服务，填写服务地址、账号实际可用的模型 ID 和 API Key。先保存，再点击“测试连接”；保存设置不等于连接已通过。

- DeepSeek 地址示例：`https://api.deepseek.com`。
- OpenAI 地址示例：`https://api.openai.com/v1`。
- 兼容服务填写其 Chat Completions API 基础地址与模型 ID。

模型 ID 以服务商账号当前支持的名称为准。测试连接会发送一次请求；普通聊天及必要任务摘要会发送至所选服务，照片和模型权重不会随聊天上传。Key 仅保留在当前服务内存，重启需重新输入；也可使用 `llm.example.env` 中列出的启动环境变量。不将 Key 填入项目说明或提交到 GitHub。

LLM 解释意图并提出受支持的下一步，程序依据真实项目状态校验参数。数据导入、训练准备和模型决定遵循同一工作流；训练与采用仍需核对确认。未配置 LLM 时，规则流程继续可用。

若连接失败，先核对页面显示的 HTTP 状态、超时、DNS、代理或 TLS 错误，再核对模型 ID、Key 权限与服务地址。无需为 LLM API 安装 PyTorch 或启动 GPU Worker。
