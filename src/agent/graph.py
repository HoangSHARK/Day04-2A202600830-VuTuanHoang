from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from src.core.llm import build_chat_model, normalize_content
from src.core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)
from src.utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


def build_system_prompt(today: str | None = None) -> str:
    current_day = today or "2026-06-01"
    return f"""
Bạn là một trợ lý ảo hỗ trợ đặt hàng các sản phẩm điện tử. Hôm nay là ngày {current_day}.
Giao tiếp hoàn toàn bằng tiếng Việt. Trả lời ngắn gọn, súc tích và bám sát kết quả.

QUY TẮC BẮT BUỘC (CRITICAL RULES):
1. Tuyệt đối không tự bịa ra thông tin sản phẩm, giá cả, khuyến mãi, tổng tiền hay đường dẫn file lưu. Chỉ sử dụng thông tin từ kết quả của các công cụ.
2. Từ chối thực hiện các yêu cầu vi phạm chính sách: không xuất hóa đơn giả, không áp dụng mã giảm giá tùy ý, không bỏ qua kiểm tra số lượng tồn kho (bypass stock), không bỏ qua catalog. Khi gặp các yêu cầu này, từ chối thẳng và KHÔNG gọi tool.
3. Trước khi gọi BẤT KỲ tool nào để xử lý đơn hàng, bạn PHẢI xác nhận có đủ TẤT CẢ các thông tin sau từ người dùng:
   - Tên khách hàng
   - Số điện thoại
   - Email
   - Địa chỉ giao hàng
   - Ít nhất một sản phẩm và số lượng muốn mua
   Nếu thiếu bất kỳ thông tin nào, hãy hỏi lại khách hàng và DỪNG LẠI (không gọi tool nào cả).
4. Quy trình xử lý đơn hàng hợp lệ BẮT BUỘC tuân theo trình tự công cụ sau:
   Bước 1: Gọi `list_products` để tìm kiếm sản phẩm.
   Bước 2: Gọi `get_product_details` với các product_id tìm được để lấy detail_token.
   Bước 3: Gọi `get_discount` với seed_hint là email của khách hàng.
   Bước 4: Gọi `calculate_order_totals` để kiểm tra tồn kho và tính tổng tiền.
   Bước 5: Gọi `save_order` để lưu đơn hàng (chỉ thực hiện khi tất cả bước trên đã hoàn tất và validation không có lỗi).
5. Chỉ lưu đơn khi bước tính toán thành công. Sau khi lưu thành công, thông báo cho khách hàng đường dẫn lưu đơn hàng.
""".strip()


def build_tools(store: OrderDataStore):
    @tool(args_schema=ListProductsInput)
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> str:
        """Search the local product catalog and return the best matching items."""
        return json.dumps(
            store.list_products(
                query=query,
                category=category,
                max_unit_price=max_unit_price,
                required_tags=required_tags,
                in_stock_only=in_stock_only,
                limit=limit,
            ),
            ensure_ascii=False,
        )

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Return exact product details for previously discovered product IDs."""
        return json.dumps(store.get_product_details(product_ids), ensure_ascii=False)

    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """Return the simulated campaign discount for the order."""
        return json.dumps(store.get_discount(seed_hint=seed_hint, customer_tier=customer_tier), ensure_ascii=False)

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(items, detail_token: str, discount_rate: float) -> str:
        """Validate stock and calculate the discounted order total."""
        return json.dumps(
            store.calculate_order_totals(items=items, detail_token=detail_token, discount_rate=discount_rate),
            ensure_ascii=False,
        )

    @tool(args_schema=SaveOrderInput)
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items,
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """Persist the final order to a local JSON file."""
        return json.dumps(
            store.save_order(
                customer_name=customer_name,
                customer_phone=customer_phone,
                customer_email=customer_email,
                shipping_address=shipping_address,
                items=items,
                detail_token=detail_token,
                discount_rate=discount_rate,
                campaign_code=campaign_code,
                customer_tier=customer_tier,
                notes=notes,
            ),
            ensure_ascii=False,
        )

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "google",
    model_name: str | None = None,
    today: str | None = None,
):
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    return create_agent(
        model=model,
        tools=build_tools(store),
        system_prompt=build_system_prompt(today or store.today),
    )


def run_agent(
    query: str,
    *,
    provider: str = "google",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    agent = build_agent(
        data_dir=data_dir,
        output_dir=output_dir,
        provider=provider,
        model_name=model_name,
        today=today,
    )
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = response["messages"] if isinstance(response, dict) else response
    tool_calls = extract_tool_calls(messages)
    saved_order, saved_order_path = extract_saved_order(tool_calls)
    return AgentResult(
        query=query,
        final_answer=extract_final_answer(messages),
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )


def extract_final_answer(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    pending: dict[str, dict[str, Any]] = {}
    records: list[ToolCallRecord] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                pending[tool_call["id"]] = {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}) or {},
                }
        elif isinstance(message, ToolMessage):
            metadata = pending.pop(message.tool_call_id, {})
            records.append(
                ToolCallRecord(
                    name=str(getattr(message, "name", None) or metadata.get("name", "")),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )

    for metadata in pending.values():
        records.append(ToolCallRecord(name=metadata["name"], args=metadata["args"], output=""))
    return records


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue
        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "saved":
            return None, None
        return payload.get("saved_order"), payload.get("path")
    return None, None
