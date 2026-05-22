import importlib
import json
import logging
import re
import requests
from typing import List

from loguru import logger
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletion

from app.config import config

_max_retries = 5
_DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
_DEPRECATED_GEMINI_MODELS = {"gemini-pro", "gemini-1.0-pro"}


def _normalize_text_response(content, llm_provider: str) -> str:
    # 不同 LLM SDK 在异常或被拦截场景下，可能返回 None、空字符串，
    # 甚至返回非字符串对象。这里统一做兜底校验，避免后续直接调用
    # `.replace()` 时抛出 `NoneType` 之类的属性错误。
    if content is None:
        raise ValueError(f"[{llm_provider}] returned empty text content")

    if not isinstance(content, str):
        raise TypeError(
            f"[{llm_provider}] returned non-text content: {type(content).__name__}"
        )

    content = content.strip()
    if not content:
        raise ValueError(f"[{llm_provider}] returned empty text content")

    return content.replace("\n", "")


def _extract_chat_completion_text(response, llm_provider: str) -> str:
    # OpenAI 兼容接口在异常场景下，可能返回没有 choices、
    # 或者 choices/message/content 为空的响应对象。
    # 这里统一做结构校验，避免出现 `NoneType is not subscriptable`
    # 这类底层属性访问错误。
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError(f"[{llm_provider}] returned empty choices")

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        raise ValueError(f"[{llm_provider}] returned empty message")

    content = getattr(message, "content", None)
    return _normalize_text_response(content, llm_provider)


def _generate_response(prompt: str) -> str:
    try:
        content = ""
        llm_provider = config.app.get("llm_provider", "openai")
        logger.info(f"llm provider: {llm_provider}")
        if llm_provider == "g4f":
            if not config.app.get("enable_g4f", False):
                raise ValueError(
                    "g4f provider is disabled by default because it relies on "
                    "reverse-engineered third-party endpoints. Set enable_g4f=true "
                    "in config.toml only if you understand and accept the security, "
                    "reliability, and legal risks."
                )

            logger.warning(
                "g4f provider is enabled. This provider may be unstable and carries "
                "supply-chain and terms-of-service risks. Prefer official providers, "
                "OpenAI-compatible APIs, LiteLLM, Ollama, or local inference for production."
            )
            try:
                g4f = importlib.import_module("g4f")
            except ImportError as e:
                raise ValueError(
                    "g4f package is not installed by default. Install the optional "
                    "dependency with `uv sync --extra g4f` only if you understand "
                    "and accept the provider risks."
                ) from e

            model_name = config.app.get("g4f_model_name", "")
            if not model_name:
                model_name = "gpt-3.5-turbo-16k-0613"
            content = g4f.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
        else:
            api_version = ""  # for azure
            if llm_provider == "moonshot":
                api_key = config.app.get("moonshot_api_key")
                model_name = config.app.get("moonshot_model_name")
                base_url = "https://api.moonshot.cn/v1"
            elif llm_provider == "ollama":
                # api_key = config.app.get("openai_api_key")
                api_key = "ollama"  # any string works but you are required to have one
                model_name = config.app.get("ollama_model_name")
                base_url = config.app.get("ollama_base_url", "")
                if not base_url:
                    base_url = "http://localhost:11434/v1"
            elif llm_provider == "openai":
                api_key = config.app.get("openai_api_key")
                model_name = config.app.get("openai_model_name")
                base_url = config.app.get("openai_base_url", "")
                if not base_url:
                    base_url = "https://api.openai.com/v1"
            elif llm_provider == "oneapi":
                api_key = config.app.get("oneapi_api_key")
                model_name = config.app.get("oneapi_model_name")
                base_url = config.app.get("oneapi_base_url", "")
            elif llm_provider == "azure":
                api_key = config.app.get("azure_api_key")
                model_name = config.app.get("azure_model_name")
                base_url = config.app.get("azure_base_url", "")
                api_version = config.app.get("azure_api_version", "2024-02-15-preview")
            elif llm_provider == "gemini":
                api_key = config.app.get("gemini_api_key")
                model_name = config.app.get("gemini_model_name")
                base_url = config.app.get("gemini_base_url", "")
                # Gemini 旧模型名已经陆续下线，这里自动兼容历史配置，
                # 避免用户沿用旧值时直接收到 404。
                if not model_name:
                    model_name = _DEFAULT_GEMINI_MODEL
                elif model_name in _DEPRECATED_GEMINI_MODELS:
                    logger.warning(
                        f"gemini model '{model_name}' is deprecated, fallback to '{_DEFAULT_GEMINI_MODEL}'"
                    )
                    model_name = _DEFAULT_GEMINI_MODEL
            elif llm_provider == "qwen":
                api_key = config.app.get("qwen_api_key")
                model_name = config.app.get("qwen_model_name")
                base_url = "***"
            elif llm_provider == "cloudflare":
                api_key = config.app.get("cloudflare_api_key")
                model_name = config.app.get("cloudflare_model_name")
                account_id = config.app.get("cloudflare_account_id")
                base_url = "***"
            elif llm_provider == "minimax":
                api_key = config.app.get("minimax_api_key")
                model_name = config.app.get("minimax_model_name")
                base_url = config.app.get("minimax_base_url", "")
                if not base_url:
                    base_url = "https://api.minimax.io/v1"
            elif llm_provider == "deepseek":
                api_key = config.app.get("deepseek_api_key")
                model_name = config.app.get("deepseek_model_name")
                base_url = config.app.get("deepseek_base_url")
                if not base_url:
                    base_url = "https://api.deepseek.com"
            elif llm_provider == "modelscope":
                api_key = config.app.get("modelscope_api_key")
                model_name = config.app.get("modelscope_model_name")
                base_url = config.app.get("modelscope_base_url")
                if not base_url:
                    base_url = "https://api-inference.modelscope.cn/v1/"
            elif llm_provider == "ernie":
                api_key = config.app.get("ernie_api_key")
                secret_key = config.app.get("ernie_secret_key")
                base_url = config.app.get("ernie_base_url")
                model_name = "***"
                if not secret_key:
                    raise ValueError(
                        f"{llm_provider}: secret_key is not set, please set it in the config.toml file."
                    )
            elif llm_provider == "pollinations":
                try:
                    base_url = config.app.get("pollinations_base_url", "")
                    if not base_url:
                        base_url = "https://text.pollinations.ai/openai"
                    model_name = config.app.get("pollinations_model_name", "openai-fast")
                   
                    # Prepare the payload
                    payload = {
                        "model": model_name,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "seed": 101  # Optional but helps with reproducibility
                    }
                    
                    # Optional parameters if configured
                    if config.app.get("pollinations_private"):
                        payload["private"] = True
                    if config.app.get("pollinations_referrer"):
                        payload["referrer"] = config.app.get("pollinations_referrer")
                    
                    headers = {
                        "Content-Type": "application/json"
                    }
                    
                    # Make the API request
                    response = requests.post(base_url, headers=headers, json=payload)
                    response.raise_for_status()
                    result = response.json()
                    
                    if result and "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0]["message"]["content"]
                        return _normalize_text_response(content, llm_provider)
                    else:
                        raise Exception(f"[{llm_provider}] returned an invalid response format")
                        
                except requests.exceptions.RequestException as e:
                    raise Exception(f"[{llm_provider}] request failed: {str(e)}")
                except Exception as e:
                    raise Exception(f"[{llm_provider}] error: {str(e)}")

            elif llm_provider == "litellm":
                model_name = config.app.get("litellm_model_name")

            if llm_provider not in ["pollinations", "ollama", "litellm"]:  # Skip validation for providers that don't require API key
                if not api_key:
                    raise ValueError(
                        f"{llm_provider}: api_key is not set, please set it in the config.toml file."
                    )
                if not model_name:
                    raise ValueError(
                        f"{llm_provider}: model_name is not set, please set it in the config.toml file."
                    )
                if not base_url and llm_provider not in ["gemini"]:
                    raise ValueError(
                        f"{llm_provider}: base_url is not set, please set it in the config.toml file."
                    )

            if llm_provider == "qwen":
                import dashscope
                from dashscope.api_entities.dashscope_response import GenerationResponse

                dashscope.api_key = api_key
                response = dashscope.Generation.call(
                    model=model_name, messages=[{"role": "user", "content": prompt}]
                )
                if response:
                    if isinstance(response, GenerationResponse):
                        status_code = response.status_code
                        if status_code != 200:
                            raise Exception(
                                f'[{llm_provider}] returned an error response: "{response}"'
                            )

                        content = response["output"]["text"]
                        return content.replace("\n", "")
                    else:
                        raise Exception(
                            f'[{llm_provider}] returned an invalid response: "{response}"'
                        )
                else:
                    raise Exception(f"[{llm_provider}] returned an empty response")

            if llm_provider == "gemini":
                import google.generativeai as genai

                if not base_url:
                    genai.configure(api_key=api_key, transport="rest")
                else:
                    genai.configure(api_key=api_key, transport="rest", client_options={'api_endpoint': base_url})

                generation_config = {
                    "temperature": 0.5,
                    "top_p": 1,
                    "top_k": 1,
                    "max_output_tokens": 2048,
                }

                safety_settings = [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_ONLY_HIGH",
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_ONLY_HIGH",
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_ONLY_HIGH",
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_ONLY_HIGH",
                    },
                ]

                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                )

                try:
                    response = model.generate_content(prompt)
                    candidates = response.candidates
                    generated_text = candidates[0].content.parts[0].text
                except (AttributeError, IndexError) as e:
                    logger.warning(
                        f"gemini returned invalid response content: {str(e)}"
                    )
                    raise ValueError(
                        f"[{llm_provider}] returned invalid response content"
                    )

                return _normalize_text_response(generated_text, llm_provider)

            if llm_provider == "cloudflare":
                response = requests.post(
                    f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_name}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a friendly assistant",
                            },
                            {"role": "user", "content": prompt},
                        ]
                    },
                )
                result = response.json()
                logger.info(result)
                return _normalize_text_response(result["result"]["response"], llm_provider)

            if llm_provider == "ernie":
                response = requests.post(
                    "https://aip.baidubce.com/oauth/2.0/token", 
                    params={
                        "grant_type": "client_credentials",
                        "client_id": api_key,
                        "client_secret": secret_key,
                    }
                )
                access_token = response.json().get("access_token")
                url = f"{base_url}?access_token={access_token}"

                payload = json.dumps(
                    {
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.5,
                        "top_p": 0.8,
                        "penalty_score": 1,
                        "disable_search": False,
                        "enable_citation": False,
                        "response_format": "text",
                    }
                )
                headers = {"Content-Type": "application/json"}

                response = requests.request(
                    "POST", url, headers=headers, data=payload
                ).json()
                return _normalize_text_response(response.get("result"), llm_provider)

            if llm_provider == "litellm":
                import litellm

                if not model_name:
                    raise ValueError(
                        f"{llm_provider}: model_name is not set, please set it in the config.toml file."
                    )

                response = litellm.completion(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    drop_params=True,
                )

                if not response:
                    raise ValueError(f"[{llm_provider}] returned empty response")
                if not getattr(response, "choices", None):
                    raise ValueError(f"[{llm_provider}] returned empty response")

                return _extract_chat_completion_text(response, llm_provider)

            if llm_provider == "azure":
                client = AzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=base_url,
                )

            if llm_provider == "modelscope":
                content = ''
                client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                )
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    extra_body={"enable_thinking": False},
                    stream=True
                )
                if response:
                    for chunk in response:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            content += delta.content
                    
                    if not content.strip():
                        raise ValueError("Empty content in stream response")
                    
                    return _normalize_text_response(content, llm_provider)
                else:
                    raise Exception(f"[{llm_provider}] returned an empty response")

            else:
                client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                )

            response = client.chat.completions.create(
                model=model_name, messages=[{"role": "user", "content": prompt}]
            )
            if response:
                if isinstance(response, ChatCompletion):
                    return _extract_chat_completion_text(response, llm_provider)
                else:
                    raise Exception(
                        f'[{llm_provider}] returned an invalid response: "{response}", please check your network '
                        f"connection and try again."
                    )
            else:
                raise Exception(
                    f"[{llm_provider}] returned an empty response, please check your network connection and try again."
                )

        return _normalize_text_response(content, llm_provider)
    except Exception as e:
        return f"Error: {str(e)}"


def generate_script(
    video_subject: str, language: str = "", paragraph_number: int = 1
) -> dict:
    prompt = f"""
# Role: Video Script Generator

## Profile:
你是一位精通地缘政治、宏观经济和社会观察的顶级自媒体商业评论员、视频主笔。你擅长用极具磁性、口语化且富有思辨性的语言，将枯燥的数据和复杂的国际新闻拆解得通俗易懂，直击底层逻辑。

## Goals:
1. 根据前几天的热搜选择1到2个话题，创作一篇结构严谨、层层剥茧、带有“看透本质”观点的深度解说文案。
2. 同时生成 2-3 个抓人标题，风格要有“没人敢聊 / 到底出了什么问题 / 成果出来了但代价也来了”这类悬念感、冲突感和传播性。
3. 文案长度目标为 3600 字左右，保证信息密度、节奏感和可听性。

## Tone & Style:
1. 采用口语化说书人节奏，多用“说实话”“各位要知道”“这个账其实不难算”“这还不算最绝的”“问题就出在这儿”等自然过渡句。
2. 如果涉及复杂数据、金融逻辑、技术术语、产业链概念或地缘政治术语，必须用老百姓能听懂的生活场景、商业常识、历史典故或身边经验做类比。
3. 保持清醒、克制、现实主义，不灌鸡汤，不喊口号，不要廉价煽情。
4. 观点要犀利，但判断必须建立在逻辑链条、结构关系和现实约束之上。

## Workflow & Structure:
请严格按照以下四步法组织正文：

### 1. 引子与事实抛出 (The Fact)
- 开头直接切入主题，例如“今天我们来聊聊……”“说实话，有几个数字挺让人意外的”。
- 迅速抛出核心事件、关键反差或震撼数据，建立现实与预期之间的落差。

### 2. 核心疑问与反常点 (The Question)
- 提出一个直击问题本质的疑问，例如“那么问题出在哪？”“为什么会这样？”“这背后到底在怕什么？”
- 引导受众意识到表面现象下的不合理之处。

### 3. 深度多维归因 (The Analysis)
- 必须使用“列举可能 / 原因拆解”的逻辑链条，比如“第一种可能……第二种可能……”或“原因大概有这么几个”。
- 必须引入至少一个底层理论框架、对比模型或解释工具，例如宏观经济周期、产权理论、技术红利分配模型、产业链控制权、地缘博弈筹码、金融杠杆传导等。
- 把孤立新闻上升到更大的时代背景、行业规律、国家博弈或资源分配逻辑。

### 4. 现实主义收尾 (The Conclusion)
- 总结事件的深远影响，点出赢家、输家、未来风险或政策约束。
- 用硬核、清醒、带有预警感或讽刺意味的句子收尾，例如“这一切不过是……”“真正的考验其实刚开始”“这场风暴很可能还没到最猛烈的时候”。

## Writing Requirements:
1. Return ONLY a valid JSON object with exactly two fields: "video_title" and "video_script".
2. "video_title" must be a JSON array of 2-3 strings, each string is one candidate title.
3. 标题必须抓人，但不能低级标题党；要有悬念、冲突、反常识感和传播力。
4. "video_script" 只包含正文，不得把标题内容重复写成正文第一行或小标题。
5. 正文不要使用 markdown、列表符号、括号小节标题、标签如“旁白”“解说词”或任何额外解释。
6. 正文必须信息密度高、逻辑递进清晰、口语化顺畅，适合直接配音。
7. 如果用户只提供一个主题词，也必须主动补全背景、利益关系、历史纵深和现实约束。
8. 正文默认写成完整长文，目标长度约 3600 字；除非素材极短到无法支撑，否则不要写得过短。
9. 使用与视频主题相同的语言；如果明确指定 language，则严格按该语言输出。
10. 不要提及提示词、四步法、分析过程，也不要解释你是如何生成内容的。

## Context:
- video subject: {video_subject}
- number of paragraphs: {paragraph_number}
""".strip()
    if language:
        prompt += f"\n- language: {language}"

    final_result = {"video_title": [], "video_script": ""}
    logger.info(f"subject: {video_subject}")

    def format_response(response):
        response = response.replace("*", "").replace("#", "")
        match = re.search(r"\{.*\}", response)
        if not match:
            raise ValueError("response is not a valid JSON object")
        payload = json.loads(match.group())
        raw_video_title = payload.get("video_title", [])
        video_script = str(payload.get("video_script", "")).strip()

        video_titles = []
        if isinstance(raw_video_title, list):
            for item in raw_video_title:
                title = re.sub(r"\s+", " ", str(item or "")).strip().strip('"\'“”‘’《》【】[]()')
                if title:
                    video_titles.append(title)
        elif isinstance(raw_video_title, str):
            title = re.sub(r"\s+", " ", raw_video_title).strip().strip('"\'“”‘’《》【】[]()')
            if title:
                video_titles.append(title)

        if not video_titles or not video_script:
            raise ValueError("response is missing video_title or video_script")

        video_script = re.sub(r"\[.*\]", "", video_script)
        video_script = re.sub(r"\(.*\)", "", video_script)
        video_script = video_script.strip()

        script_lines = [line.strip() for line in video_script.splitlines() if line.strip()]
        if script_lines and script_lines[0] in video_titles:
            script_lines = script_lines[1:]
        video_script = "\n\n".join(script_lines) if script_lines else video_script

        return {"video_title": video_titles, "video_script": video_script.strip()}

    for i in range(_max_retries):
        try:
            response = _generate_response(prompt=prompt)
            if response:
                final_result = format_response(response)
            else:
                logging.error("gpt returned an empty response")

            # g4f may return an error message
            if final_result["video_script"] and "当日额度已消耗完" in final_result["video_script"]:
                raise ValueError(final_result["video_script"])

            if final_result["video_title"] and final_result["video_script"]:
                break
        except Exception as e:
            logger.error(f"failed to generate script: {e}")

        if i < _max_retries:
            logger.warning(f"failed to generate video script, trying again... {i + 1}")
    if "Error: " in final_result["video_script"]:
        logger.error(f"failed to generate video script: {final_result['video_script']}")
    else:
        logger.success(
            "completed titles: "
            f"{json.dumps(final_result['video_title'], ensure_ascii=False)}\ncompleted script: \n{final_result['video_script']}"
        )
    return final_result


def generate_terms(video_subject: str, video_script: str, amount: int = 20) -> List[str]:
    prompt = f"""
# Role: Video Search Terms Generator

## Profile:
你是一位精通地缘政治、宏观经济和社会观察的顶级视频主笔，同时也是非常懂镜头语言的策划编辑。你能把一篇充满洞察的商业评论、国际新闻解读或宏观分析文案，拆成适合剪辑的视觉线索。

## Goals:
1. 深度理解输入主题和正文里的叙事结构、事实锚点、反常识点、利益关系和时代背景。
2. 按照“引子与事实抛出 -> 核心疑问与反常点 -> 深度多维归因 -> 现实主义收尾”的四步法，在脑中还原这篇评论的视觉节奏。
3. 生成 {amount} 个适合 stock footage / B-roll 检索的高质量英文搜索词。

## Visual Conversion Principles:
1. 搜索词必须覆盖人物、产业、城市、港口、工厂、金融市场、政策场景、贸易物流、会议博弈、社会情绪、日常生活等多个视觉层级。
2. 既要有微观镜头，也要有宏观镜头。微观比如工厂流水线、超市货架、家庭消费、失业招聘、码头装箱；宏观比如央行、股市、航运、能源、边境、外交会谈、城市天际线。
3. 如果正文里出现复杂概念，比如关税、汇率、产能外溢、金融制裁、技术封锁、能源安全、房地产周期、供应链重组，要把这些概念翻译成“能拍出来”的画面。
4. 优先选择具体、可视化、易检索的英文短语，不要停留在抽象概念。

## Constraints for Output:
1. Return ONLY a valid JSON array of strings.
2. Do NOT output any analysis, explanation, numbering, or hidden script.
3. Each search term should preferably be 2-4 English words; allow longer only when needed for precision.
4. Every search term must correspond to something that can realistically appear in video footage.
5. Search terms must jointly覆盖事实层、问题层、归因层、结论层的视觉信息，而不是只围绕一个场景重复改写。
6. 避免 generic terms like "economy", "business", "meeting", "city" unless they are anchored with context.
7. 当主题涉及中国、美国、欧洲、中东、俄罗斯、东南亚等特定区域时，优先把地域信息体现在搜索词里。
8. Search terms must be in English only.

## Output Example:
["container port china", "factory assembly line", "currency exchange board", "oil tanker terminal", "central bank press conference"]

## Context:
### Video Subject
{video_subject}

### Video Script
{video_script}

## Execution Instruction:
先理解这篇稿子里最重要的事实、反差、问题、归因和收尾判断，再把这些抽象判断拆成可以被镜头表达的画面元素。最终输出 {amount} 个高质量、可检索、去抽象化的英文搜索词，只输出 JSON 数组。
""".strip()

    logger.info(f"subject: {video_subject}")

    search_terms = []
    response = ""
    for i in range(_max_retries):
        try:
            response = _generate_response(prompt)
            if "Error: " in response:
                logger.error(f"failed to generate video script: {response}")
                return response
            search_terms = json.loads(response)
            if not isinstance(search_terms, list) or not all(
                isinstance(term, str) for term in search_terms
            ):
                logger.error("response is not a list of strings.")
                continue

        except Exception as e:
            logger.warning(f"failed to generate video terms: {str(e)}")
            if response:
                match = re.search(r"\[.*]", response)
                if match:
                    try:
                        search_terms = json.loads(match.group())
                    except Exception as e:
                        # 这里保留重试流程，但必须记录 LLM 返回的非标准 JSON，
                        # 否则后续排查搜索词为空时无法定位
                        # 是模型格式问题还是解析逻辑问题。
                        logger.warning(f"failed to generate video terms: {str(e)}")

        if search_terms and len(search_terms) > 0:
            break
        if i < _max_retries:
            logger.warning(f"failed to generate video terms, trying again... {i + 1}")

    logger.success(f"completed: \n{search_terms}")
    return search_terms


if __name__ == "__main__":
    video_subject = "生命的意义是什么"
    result = generate_script(
        video_subject=video_subject, language="zh-CN", paragraph_number=1
    )
    print("######################")
    print("\n".join(result["video_title"]))
    print("######################")
    print(result["video_script"])
    search_terms = generate_terms(
        video_subject=video_subject, video_script=result["video_script"], amount=20
    )
    print("######################")
    print(search_terms)
    
