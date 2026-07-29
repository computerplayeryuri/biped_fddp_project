#!/usr/bin/env python3
"""Generate a detailed Chinese guide to every formal figure in the thesis.

The document reuses the exact figure assets already included by the thesis.
The two figures authored directly in TikZ are reconstructed from their current
LaTeX source so the guide remains self-contained without requiring TeX.
"""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Iterable

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
THESIS_ROOT = ROOT / "docs" / "overleaf" / "thesis_6a57ac33"
FIGURE_ROOT = THESIS_ROOT / "figures"
OUTPUT_DIR = ROOT / "output" / "pdf"
TMP_DIR = ROOT / "tmp" / "pdfs"
OUTPUT_PDF = OUTPUT_DIR / "thesis_all_figures_detailed_explanation_zh.pdf"

FONT_REGULAR = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")
FONT_NAME = "MicrosoftYaHei"
FONT_BOLD_NAME = "MicrosoftYaHeiBold"

PAGE_WIDTH, PAGE_HEIGHT = A4
LEFT_MARGIN = 18 * mm
RIGHT_MARGIN = 18 * mm
TOP_MARGIN = 18 * mm
BOTTOM_MARGIN = 17 * mm
CONTENT_WIDTH = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN

NAVY = colors.HexColor("#16324F")
BLUE = colors.HexColor("#1769AA")
GREEN = colors.HexColor("#167D5A")
ORANGE = colors.HexColor("#C46A1B")
RED = colors.HexColor("#A63D40")
LIGHT_BLUE = colors.HexColor("#EAF3FA")
LIGHT_GREEN = colors.HexColor("#EAF6F1")
LIGHT_ORANGE = colors.HexColor("#FFF4E8")
LIGHT_RED = colors.HexColor("#FBEDEE")
LIGHT_GREY = colors.HexColor("#F4F6F8")
MID_GREY = colors.HexColor("#66727D")
TEXT = colors.HexColor("#1C252C")


def register_fonts() -> None:
    if not FONT_REGULAR.is_file() or not FONT_BOLD.is_file():
        raise FileNotFoundError("Microsoft YaHei fonts are required")
    pdfmetrics.registerFont(TTFont(FONT_NAME, str(FONT_REGULAR)))
    pdfmetrics.registerFont(TTFont(FONT_BOLD_NAME, str(FONT_BOLD)))
    pdfmetrics.registerFontFamily(
        FONT_NAME,
        normal=FONT_NAME,
        bold=FONT_BOLD_NAME,
        italic=FONT_NAME,
        boldItalic=FONT_BOLD_NAME,
    )


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCN",
            parent=base["Title"],
            fontName=FONT_BOLD_NAME,
            fontSize=25,
            leading=34,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=12,
            wordWrap="CJK",
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCN",
            parent=base["Normal"],
            fontName=FONT_NAME,
            fontSize=11,
            leading=18,
            textColor=MID_GREY,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
        "h1": ParagraphStyle(
            "Heading1CN",
            parent=base["Heading1"],
            fontName=FONT_BOLD_NAME,
            fontSize=18,
            leading=25,
            textColor=NAVY,
            spaceBefore=3,
            spaceAfter=8,
            wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "Heading2CN",
            parent=base["Heading2"],
            fontName=FONT_BOLD_NAME,
            fontSize=12.5,
            leading=18,
            textColor=BLUE,
            spaceBefore=8,
            spaceAfter=4,
            wordWrap="CJK",
        ),
        "body": ParagraphStyle(
            "BodyCN",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=9.5,
            leading=15.2,
            textColor=TEXT,
            alignment=TA_LEFT,
            wordWrap="CJK",
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "SmallCN",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=7.8,
            leading=11.5,
            textColor=MID_GREY,
            alignment=TA_LEFT,
            wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "CaptionCN",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=8,
            leading=12,
            textColor=MID_GREY,
            alignment=TA_CENTER,
            wordWrap="CJK",
            spaceBefore=3,
            spaceAfter=7,
        ),
        "box": ParagraphStyle(
            "BoxCN",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=9.2,
            leading=14.8,
            textColor=TEXT,
            alignment=TA_LEFT,
            wordWrap="CJK",
        ),
        "bullet": ParagraphStyle(
            "BulletCN",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=9.1,
            leading=14.3,
            textColor=TEXT,
            leftIndent=2 * mm,
            firstLineIndent=0,
            wordWrap="CJK",
        ),
        "toc": ParagraphStyle(
            "TocCN",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=9.3,
            leading=15,
            textColor=TEXT,
            wordWrap="CJK",
        ),
    }


def clean(value: str) -> str:
    return html.escape(value).replace("\n", "<br/>")


def paragraph(value: str, style) -> Paragraph:
    return Paragraph(clean(value), style)


def rich_paragraph(value: str, style) -> Paragraph:
    return Paragraph(value, style)


def info_box(title: str, text: str, style, background, accent):
    content = [
        rich_paragraph(
            f'<font name="{FONT_BOLD_NAME}" color="{accent.hexval()}">{clean(title)}</font><br/>'
            f"{clean(text)}",
            style,
        )
    ]
    table = Table([[content]], colWidths=[CONTENT_WIDTH], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.5, accent),
                ("LINEBEFORE", (0, 0), (0, -1), 3.2, accent),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def bullets(items: Iterable[str], style) -> ListFlowable:
    return ListFlowable(
        [ListItem(paragraph(item, style), leftIndent=4 * mm) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=6 * mm,
        bulletFontName=FONT_NAME,
        bulletFontSize=7,
        bulletColor=BLUE,
        spaceAfter=4,
    )


def scaled_image(path: Path, max_width=CONTENT_WIDTH, max_height=128 * mm) -> Image:
    if not path.is_file():
        raise FileNotFoundError(path)
    with PILImage.open(path) as source:
        width, height = source.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def pil_font(size: int, bold: bool = False):
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size=size)


def wrapped_lines(draw: ImageDraw.ImageDraw, text: str, font, max_width: int):
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = ""
        for character in raw_line:
            candidate = line + character
            if line and draw.textlength(candidate, font=font) > max_width:
                lines.append(line)
                line = character
            else:
                line = candidate
        lines.append(line)
    return lines


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font,
    fill="#1C252C",
    spacing=8,
):
    x0, y0, x1, y1 = box
    lines = wrapped_lines(draw, text, font, x1 - x0 - 28)
    line_heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line or "中", font=font)
        line_heights.append(bbox[3] - bbox[1])
    total = sum(line_heights) + spacing * max(0, len(lines) - 1)
    y = y0 + (y1 - y0 - total) / 2
    for line, height in zip(lines, line_heights):
        width = draw.textlength(line, font=font)
        draw.text(((x0 + x1 - width) / 2, y), line, font=font, fill=fill)
        y += height + spacing


def rounded_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    outline: str,
    fill: str,
    font,
    radius=18,
    width=4,
):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
    draw_centered_text(draw, box, text, font)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    fill="#27343D",
    width=5,
):
    draw.line([start, end], fill=fill, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 18
    spread = 0.55
    points = [
        end,
        (
            end[0] - length * math.cos(angle - spread),
            end[1] - length * math.sin(angle - spread),
        ),
        (
            end[0] - length * math.cos(angle + spread),
            end[1] - length * math.sin(angle + spread),
        ),
    ]
    draw.polygon(points, fill=fill)


def reconstruct_positioning(path: Path) -> None:
    canvas = PILImage.new("RGB", (2400, 1350), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = pil_font(46, True)
    axis_font = pil_font(30, True)
    box_font = pil_font(29)
    label_font = pil_font(27)
    draw.text((1200, 50), "研究定位图（依据论文 TikZ 源码重绘）", font=title_font, fill="#16324F", anchor="ma")
    origin = (180, 1140)
    x_end = (2240, 1140)
    y_end = (180, 170)
    arrow(draw, origin, x_end, width=6)
    arrow(draw, origin, y_end, width=6)
    for y in (900, 660, 420):
        draw.line([(180, y), (2200, y)], fill="#D7DDE2", width=2)
    draw.text((1220, 1210), "验证成熟度与证据范围：解析/仿真 -> 真实硬件/地形", font=axis_font, fill="#1C252C", anchor="ma")
    vertical = PILImage.new("RGBA", (900, 80), (255, 255, 255, 0))
    vdraw = ImageDraw.Draw(vertical)
    vdraw.text((450, 40), "模型与控制复杂度：经验/降阶 -> 全身/高带宽", font=axis_font, fill="#1C252C", anchor="mm")
    vertical = vertical.rotate(90, expand=True)
    canvas.paste(vertical, (20, 280), vertical)

    rounded_box(draw, (240, 850, 650, 1015), "LIPM/预览控制\n低阶预测模型\n仿真证据", outline="#3467A5", fill="#FFFFFF", font=box_font)
    rounded_box(draw, (560, 600, 980, 775), "降阶 MPC\nALIP/凸模型\n显式约束", outline="#3467A5", fill="#FFFFFF", font=box_font)
    rounded_box(draw, (420, 245, 890, 425), "离线全身轨迹优化\n完整刚体模型", outline="#3467A5", fill="#FFFFFF", font=box_font)
    rounded_box(draw, (1030, 515, 1545, 750), "本文\n离线 FDDP + FIR-QP-MPC\n位置命令接口\n仿真验证", outline="#167D5A", fill="#EAF6F1", font=pil_font(31, True), width=6)
    rounded_box(draw, (1660, 260, 2135, 445), "硬件全身控制\n高带宽驱动\n接触力控制权限", outline="#704C91", fill="#FFFFFF", font=box_font)
    rounded_box(draw, (1770, 770, 2210, 950), "硬件运动\n地形与推力测试\n物理证据", outline="#704C91", fill="#FFFFFF", font=box_font)

    boundary_x = 1585
    draw.line([(boundary_x, 210), (boundary_x, 1080)], fill="#7A8288", width=5)
    for y in range(210, 1080, 28):
        draw.line([(boundary_x, y), (boundary_x, min(y + 14, 1080))], fill="white", width=5)
    draw.text((boundary_x, 145), "当前论文证据边界", font=label_font, fill="#66727D", anchor="mm")
    canvas.save(path, quality=95)


def reconstruct_architecture(path: Path) -> None:
    canvas = PILImage.new("RGB", (2600, 1450), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = pil_font(44, True)
    lane_font = pil_font(32, True)
    box_font = pil_font(27)
    draw.text((1300, 45), "三层系统架构（依据论文 TikZ 源码重绘）", font=title_font, fill="#16324F", anchor="ma")

    lanes = [
        (80, 140, 2520, 470, "#F4F6F8", "离线规划层"),
        (80, 515, 2520, 970, "#F4F6F8", "在线执行层"),
        (80, 1020, 2520, 1370, "#F4F6F8", "证据与验证层"),
    ]
    for x0, y0, x1, y1, fill, label in lanes:
        draw.rounded_rectangle((x0, y0, x1, y1), radius=24, fill=fill, outline="#AAB4BC", width=3)
        draw.text((x0 + 35, y0 + 20), label, font=lane_font, fill="#16324F")

    boxes = {
        "model": (150, 245, 560, 390, "机器人描述\nURDF 与 MuJoCo"),
        "fddp": (820, 245, 1250, 390, "离线 FDDP\n接触约束轨迹规划"),
        "ref": (1540, 220, 2070, 410, "已接受参考\nq_ref、接触时序\ntau_ref 仅用于分析"),
        "controller": (520, 655, 960, 825, "在线层\nC0 / C1 / C2\n有界修正 Δq"),
        "command": (1140, 655, 1640, 825, "关节位置命令\nq_cmd = q_ref + Δq"),
        "plant": (1820, 655, 2350, 825, "MuJoCo 执行器代理\n未来舵机路径"),
        "feedback": (520, 850, 960, 940, "实测俯仰与接触反馈"),
        "sensors": (1820, 850, 2350, 940, "被控对象状态与传感器"),
        "reports": (180, 1165, 640, 1315, "证据产物\n表格、图片与报告"),
        "gates": (1010, 1165, 1500, 1315, "接受门\n规划、接触、回放与控制"),
        "logs": (1880, 1165, 2350, 1315, "记录信号\n状态、接触与时序"),
    }
    for key, (x0, y0, x1, y1, text) in boxes.items():
        rounded_box(draw, (x0, y0, x1, y1), text, outline="#1C252C", fill="#FFFFFF", font=box_font)

    def right(key):
        x0, y0, x1, y1, _ = boxes[key]
        return x1, (y0 + y1) // 2

    def left(key):
        x0, y0, x1, y1, _ = boxes[key]
        return x0, (y0 + y1) // 2

    def top(key):
        x0, y0, x1, y1, _ = boxes[key]
        return (x0 + x1) // 2, y0

    def bottom(key):
        x0, y0, x1, y1, _ = boxes[key]
        return (x0 + x1) // 2, y1

    arrow(draw, right("model"), left("fddp"))
    arrow(draw, right("fddp"), left("ref"))
    arrow(draw, bottom("ref"), top("controller"))
    arrow(draw, right("controller"), left("command"))
    arrow(draw, right("command"), left("plant"))
    arrow(draw, bottom("plant"), top("sensors"))
    arrow(draw, left("sensors"), right("feedback"))
    arrow(draw, top("feedback"), bottom("controller"))
    arrow(draw, bottom("sensors"), top("logs"))
    arrow(draw, left("logs"), right("gates"))
    arrow(draw, left("gates"), right("reports"))
    canvas.save(path, quality=95)


FIGURES = [
    {
        "number": "图2.1",
        "title": "研究定位：复杂度与验证成熟度",
        "chapter": "第2章 文献综述与研究空白",
        "image": "reconstructed_positioning.png",
        "reconstructed": True,
        "caption": "按照模型/控制复杂度和验证成熟度对本文进行定性定位。绿色框表示本文把离线全身规划与低维在线位置修正集成起来；虚线表示当前论文仅限仿真的证据边界。该图用于区分方法类别和证据等级，而不是为不同机器人或算法排名。",
        "question": "这张图不是展示实验结果，而是在回答：本项目相对于降阶MPC、离线全身优化和硬件全身控制，方法上位于哪里，证据又走到了哪里？",
        "read": [
            "横轴表示验证成熟度和证据范围：左侧偏解析与仿真，右侧偏真实硬件、真实地形和物理扰动证据。",
            "纵轴表示模型与控制复杂度：下方偏经验模型或降阶模型，上方偏完整刚体、全身控制与高带宽力/力矩权限。",
            "绿色框是本文：离线部分采用全身FDDP，在线部分采用低维FIR-QP-MPC，最终输出是关节位置命令，当前验证停留在仿真。",
            "竖向虚线是证据边界。它提醒读者：方法复杂度可以较高，但没有硬件数据就不能把证据位置画到虚线右侧。",
            "LIPM/预览控制、ALIP/降阶MPC、离线全身优化、硬件全身控制和硬件运动测试是类别锚点，不是排行榜。",
        ],
        "symbols": [
            "FDDP：离线非线性最优控制求解器，用完整刚体与接触模型生成名义轨迹。",
            "FIR-QP-MPC：使用有限冲激响应模型形成小型二次规划，每20 ms给出有界位置修正。",
            "位置命令接口：硬件方向的最终输出是 q_cmd，而不是规划力矩 tau_ref。",
        ],
        "conclusion": "论文的贡献位置不是“发明一种比所有方法更高级的控制器”，而是把离线全身规划、低维在线位置修正、相位同步和分层证据链集成到一台低成本平面双足机器人上。",
        "boundary": "图中坐标没有数值刻度，方框位置是定性判断。它不能用来证明本文优于某篇论文，也不能证明已经达到硬件成熟度。",
        "talk": "先看纵轴：我的离线规划用了全身刚体模型，所以不是简单LIPM；再看横轴：我完成的是仿真证据链，还没有跨过硬件边界。因此我的工作重点是方法接口与可追溯验证，而不是声称完成了实物行走。",
        "trace": "来源：chapters_zh_full/chapter2.tex 中的 TikZ 图；本PDF依据该源码重绘。",
        "theory": "关联文献类别：LIPM/预览控制、约束MPC、FDDP/Crocoddyl、位置控制人形平台与仿真验证方法。",
    },
    {
        "number": "图3.1",
        "title": "三层系统架构：规划、执行、证据",
        "chapter": "第3章 系统建模与软件架构",
        "image": "reconstructed_architecture.png",
        "reconstructed": True,
        "caption": "三层系统架构。离线FDDP生成被接受的名义参考；在线层根据参考和反馈形成有界关节位置命令；被控对象的记录信号再由独立证据门审核。",
        "question": "这张图回答整套系统由哪些部分组成，以及规划输出如何经过控制器、执行器代理、被控对象和证据门形成闭环。",
        "read": [
            "离线规划层：URDF和MuJoCo描述机器人；Pinocchio/Crocoddyl中的FDDP生成 q_ref、速度/状态、接触时序以及仅供诊断的 tau_ref。",
            "在线执行层：C0直接回放参考；C1加入俯仰修正；C2再加入接触相位同步。三者最终都形成 q_cmd = q_ref + Δq。",
            "执行器代理表示延迟、一阶响应和速度限制，再由MuJoCo推进机器人动力学。",
            "传感器/状态信号返回控制器形成反馈，同时进入日志。仿真中主要使用MuJoCo状态与接触传感器；硬件阶段还需要真实状态估计和通信驱动。",
            "证据与验证层不反向“帮助”轨迹通过，而是独立读取日志，按规划、接触、回放、控制、鲁棒性与计时门生成证据。",
        ],
        "symbols": [
            "q_ref：离线FDDP导出的名义关节位置参考。",
            "Δq：C1/C2在线层计算的有界关节位置修正；C0中 Δq = 0。",
            "q_cmd：实际送入执行器代理或未来舵机运行时的关节位置命令。",
            "tau_ref：FDDP动力学优化中的力矩序列，只用于规划诊断，不是当前硬件命令。",
        ],
        "conclusion": "架构把“求出轨迹”“跟踪轨迹”“在线修正”“判断是否通过”分开，防止把优化器收敛直接写成机器人已经稳定行走。",
        "boundary": "图中硬件路径仍是未来路径。当前闭环使用MuJoCo真值/传感器信号，尚未完成真实编码器、IMU、足底接触、总线延迟和舵机动态的联合验证。",
        "talk": "这张图是全篇总图。FDDP只负责离线参考，控制器负责根据反馈做小修正，执行器代理和MuJoCo负责模拟执行，最后由证据门判断。最重要的接口是 tau_ref 不下发，真正下发的是 q_cmd。",
        "trace": "来源：chapters_zh_full/chapter3.tex 中的 TikZ 图；模型入口为 model/biped_v1.urdf 与 model/biped_planar.xml。",
        "theory": "理论分工：最优控制负责名义运动，反馈控制负责误差抑制，状态/接触观测提供闭环信息，验证层负责限定结论范围。",
    },
    {
        "number": "图4.1",
        "title": "标准FDDP核心与项目外层工程流程",
        "chapter": "第4章 离线FDDP轨迹生成",
        "image": "fddp_standard_vs_project_flow_corrected_en.png",
        "caption": "标准FDDP内部求解与项目外层流程的对比。右侧是围绕FDDP的工程流程：周期种子、连续化、接受门、导出和独立MuJoCo验证，而不是替代FDDP的新优化算法。",
        "question": "左边解释FDDP一次求解内部怎样迭代；右边解释本项目怎样多次调用FDDP并通过外层门控，把一个求解器变成可验证的十步轨迹生成流程。",
        "read": [
            "左1：定义有限时域非线性最优控制问题，目标是运行代价与终端代价之和，约束是离散动力学 x_(k+1)=f_k(x_k,u_k)。",
            "左2-3：给定名义状态/控制序列并计算代价和动力学连续性缺陷 g_k。FDDP允许初始节点之间暂时不连续。",
            "左4：反向传递建立局部二次模型并进行Riccati递推，得到前馈方向 k_k 与反馈增益 K_k。",
            "左5：前向滚动与线搜索测试步长；候选若改善综合目标则接受，否则缩小步长并重试。",
            "左6：当停止量和缺陷足够小，输出 x*、u*和局部反馈信息。这里输出的是一次优化问题的解。",
            "右1-2：先固定模型、命令边界和接触时序 DS -> LSS -> DS -> RSS -> DS，再构造状态、CoM、足端、正则化与关节安全代价。",
            "右3：通过周期两步种子和七阶段同伦逐步加强困难的边界条件。七阶段是七个优化问题，不是七次内部迭代。",
            "右4：用前一阶段解热启动更长问题，依次扩展为2、4、6、10步；最终 N=1000、Δt=0.01 s、T=10 s。",
            "右5-6：规划门和接触门通过后才导出参考，再用独立MuJoCo回放以及C0/C1/C2验证。",
        ],
        "symbols": [
            "x_k=(q_k,v_k)：第k个节点的构型与速度状态；u_k=tau_k：规划模型中的六维关节力矩。",
            "g_k=difference(x_(k+1), f_k(x_k,u_k))：相邻多重射击节点的动力学缺陷。g_k=0表示连续。",
            "δu_k=k_k+K_k δx_k：局部控制更新；k_k是前馈搜索方向，K_k是局部状态反馈。",
            "r_x≈0：一个平移步态周期后的状态接缝残差；r_u≈0：周期末端与开端控制残差。",
            "七阶段顺序：先把终端状态权重提高为1、10、100、1000，再保持终端权重并把边界控制权重提高为1、10、30。",
        ],
        "conclusion": "项目的核心优化器仍是标准FDDP；项目贡献在于如何构造、渐进求解、筛选、导出并独立验证FDDP轨迹。",
        "boundary": "七阶段、具体权重、2-4-6-10步延长顺序和接受门是本项目工程设置，不是FDDP理论规定。规划通过也不等于MuJoCo回放、鲁棒性或硬件通过。",
        "talk": "左边是算法，右边是工程。FDDP内部不断做反向传递和前向线搜索；我的外层先把两步周期问题变得可解，再热启动到十步，最后必须经过规划门、接触门和独立MuJoCo验证。",
        "trace": "图源：figures/fddp_standard_vs_project_flow_corrected_en.drawio/png。关键代码：fddp/05a_solve_fddp_periodic_cycle.py 与 fddp/05b_solve_fddp_multistep_continuation.py。",
        "theory": "理论依据：DDP、Gauss-Newton shooting、多重射击与Crocoddyl/FDDP文献。项目外层流程不应被称为新的FDDP算法。",
    },
    {
        "number": "图4.2",
        "title": "C0：固定相位的十步参考回放基线",
        "chapter": "第4章 C0执行器感知回放基线",
        "image": "c0_reference_only_current_en.png",
        "caption": "作为控制器消融统一基线的C0仅参考信号流。",
        "question": "C0要回答：不使用高层俯仰反馈修正、也不修正接触相位时，被接受的FDDP位置参考通过同一个执行器代理能表现到什么程度？",
        "read": [
            "加载十步FDDP参考：时间、q_ref、dq_ref、state_ref、相位与左右脚接触元数据；tau_ref只保留作分析。",
            "运动相位固定为墙钟时间 t，不根据观测接触改变。命令采样使用 t_cmd=clip(t+0.030,0,T)，即30 ms参考超前。",
            "q_nom=q_ref(t_cmd)；俯仰参考不是FDDP直接给出的理想姿态，而是被接受C0 MuJoCo回放的 theta_baseline(t)。",
            "不施加高层修正：alpha_k=0、Δq_k=0。q_raw=q_nom，再通过SafetyLimiter。",
            "执行器/延迟代理施加位置动态和速率限制，然后写入data.ctrl并推进MuJoCo。",
            "在同一运行中注入俯仰扰动并记录 e_theta、IAE、峰值、身高、接触、关节跟踪等指标。",
        ],
        "symbols": [
            "t_cmd：考虑30 ms参考超前后的采样时刻。",
            "theta_baseline(t)：C0名义MuJoCo俯仰基线，后续C1/C2用它定义相对误差。",
            "alpha：MPC的标量修正自由度；C0中恒为0。",
            "SafetyLimiter：关节位置、变化率与可行区间限制，不是FIR-QP反馈器。",
        ],
        "conclusion": "C0提供公平对照：C0、C1和C2共享同一参考、同一MuJoCo模型、同一执行器代理、同一安全限制和同一日志门，差异只来自新增控制层。",
        "boundary": "C0不是“整台机器人完全没有控制”。执行器内部位置环由代理表示，SafetyLimiter仍在；准确说法是“没有高层FIR-QP修正和接触相位同步”。",
        "talk": "C0就是基线。它只把FDDP位置参考向前预取30 ms并经过安全限幅和执行器代理，不使用俯仰反馈，也不改相位。后面C1和C2的收益都必须相对它比较。",
        "trace": "图源：figures/c0_reference_only_current_en.drawio/png。主入口：simulation/03u_replay_fddp_multistep_mujoco.py。",
        "theory": "实验设计依据是控制器消融：保持参考与被控对象不变，只逐层增加反馈功能。",
    },
    {
        "number": "图4.3",
        "title": "标准MPC与本项目FIR-QP-MPC",
        "chapter": "第4章 FIR-QP-MPC在线修正层",
        "image": "mpc_standard_vs_project_current_en.png",
        "caption": "一般滚动时域MPC与项目实现流程的比较。项目每20 ms使用辨识的FIR/AR模型求解标量箱约束QP，并不在线求解完整多体FDDP问题。",
        "question": "这张图回答：一般MPC的测量-预测-优化-执行循环，在本项目中分别由哪些具体数据、模型、决策变量和位置命令实现？",
        "read": [
            "左侧标准MPC：读取状态估计 x_hat；更新参考与约束；在预测时域内计算模型响应；构造跟踪、控制努力与终端代价；求解QP/NLP；只施加第一步输入；下一采样重新求解。",
            "右侧离线产物：十步q_ref(t)、C0俯仰基线theta_base(t)、FIR系数h_i和AR系数a_j。",
            "在线测量包括当前俯仰theta_k、关节状态及足部接触；接触事件用于更新有界相位偏移。",
            "墙钟俯仰误差 e_k=theta_k-theta_base(t)。已知历史命令响应 sum(h_i alpha_(k-i))从误差中分离，剩余量交给AR自由响应模型。",
            "预测写成 E=E_free+S A。E_free由AR状态与过去命令产生；S是FIR卷积矩阵；A是未来12步alpha序列。",
            "QP权衡预测误差、修正幅值和修正变化率，并施加幅值、变化率和关节可行区间。每次只使用A*的第一项alpha_k。",
            "相位同步只替换下一事件腿的参考采样时刻；俯仰基线仍使用墙钟时间，防止同步器重写恢复目标。",
            "最终 q_cmd=SafetyLimiter(q_ref,hybrid+G alpha_k)，循环周期20 ms。",
        ],
        "symbols": [
            "E：未来12步预测俯仰误差向量；E_free：无未来新修正时的预测。",
            "S：由FIR系数构成的卷积/动态矩阵；A=[alpha_k,...,alpha_(k+N-1)]。",
            "G：把一个标量alpha协调分配到六个髋-膝-踝关节的映射。",
            "QP与NLP：QP目标与约束为二次/线性结构；NLP允许一般非线性。当前在线层求解小型QP。",
        ],
        "conclusion": "本项目保留了MPC的滚动时域思想，但用局部辨识的FIR/AR模型替代在线全身动力学，用一个标量修正序列替代全身控制序列，因此能在20 ms预算内运行。",
        "boundary": "它不是全身非线性MPC，也不是在线FDDP。模型来自MuJoCo局部脉冲响应，不能直接解释成真实ST3215或所有姿态下的全局动力学。",
        "talk": "左边每一栏在右边都有对应：状态估计对应俯仰和接触测量；预测模型对应FIR加AR；优化变量对应未来alpha序列；第一步输入通过G变成六关节位置修正；下一周期重新测量再求解。",
        "trace": "图源：figures/mpc_standard_vs_project_current_en.drawio/png。核心实现：mpc/multistep_controller.py 及十步C1/C2验证入口。",
        "theory": "理论依据：滚动时域MPC、FIR模型预测控制和线性系统辨识；标量通道、G和混合腿参考属于本项目实现。",
    },
    {
        "number": "图4.4",
        "title": "C1：固定相位FIR-QP俯仰恢复",
        "chapter": "第4章 C1控制器",
        "image": "c1_fir_qp_pitch_recovery_current_en.png",
        "caption": "C1 FIR-QP俯仰恢复流程。QP输出有界标量修正，通过G映射为六关节位置命令。",
        "question": "C1要回答：保持接触相位固定时，能否用一个快速、受约束的在线修正层减小相对于C0基线的俯仰误差？",
        "read": [
            "离线辨识：在C0回放上施加正负alpha脉冲，记录根部俯仰偏差，得到长度M=23的FIR核h。",
            "自然响应模型：用三阶AR模型表示不再施加新命令时误差怎样自行衰减；必要时收缩谱半径，避免预测不稳定。",
            "在线误差：e_k=theta_meas,k-theta_baseline(t)，其中基线始终按墙钟时间读取。",
            "已知历史作用分离：d_k=e_k-sum(h_i alpha_(k-i))，避免把过去命令尚未消失的延迟响应误认为新的自然扰动。",
            "构造12步、0.24 s预测 E=E_free+S A，并求解跟踪误差、幅值和变化率三项代价。",
            "正常权重(q,r,s)=(1500,10,25)；当|e_k|超过3 deg时使用恢复权重(2500,5,10)，更强调纠错并减少对幅值/变化率的惩罚。",
            "约束包括 |alpha|<=5 deg、每20 ms变化不超过1 deg，以及由关节限位推导的alpha可行区间。求解器执行120次投影梯度迭代。",
            "只应用第一步alpha_k，再由 Δq=G alpha_k 映射到六关节，并形成 q_cmd=SafetyLimiter(q_ref(t+0.030)+G alpha_k)。",
        ],
        "symbols": [
            "h_i：第i个FIR系数，表示过去一个alpha脉冲在当前采样对俯仰的延迟贡献。",
            "a_j：AR系数，当前三阶拟合为[1.8562,-1.1130,0.14094]；拟合谱半径0.9083。",
            "G=[0.27,0.062,-0.228,0.27,0.062,-0.228]^T：六关节协调修正方向；正负号表示同一alpha下不同关节的转向。",
            "q：预测俯仰误差权重；r：alpha幅值权重；s：相邻alpha变化率权重。",
            "IAE=integral |e_theta(t)| dt：完整运行期间绝对俯仰误差的累计面积。",
        ],
        "conclusion": "C1把局部俯仰恢复问题压缩为一个小型标量QP，并在保持关节位置命令接口的前提下提供闭环修正。",
        "boundary": "G不是由通用双足理论直接给出的六个数，而是项目在当前姿态和模型附近通过控制权限标定选出的方向。FIR/AR只在辨识激励和附近工作区间得到验证。",
        "talk": "C1先用FIR记住命令的延迟作用，再用AR预测自然误差，用QP决定未来12步alpha。实际只执行第一步，并通过G同时修正六个关节。它改善俯仰，但完全不改接触相位。",
        "trace": "图源：figures/c1_fir_qp_pitch_recovery_current_en.drawio/png。辨识入口：mpc/19_identify_fddp_multistep_fir.py；控制器：mpc/multistep_controller.py。",
        "theory": "理论依据：FIR系统辨识、AR自由响应、受约束滚动时域优化；恢复阈值、权重、G和投影梯度次数是项目调参。",
    },
    {
        "number": "图4.5",
        "title": "C2：接触事件相位同步加C1俯仰恢复",
        "chapter": "第4章 C2控制器",
        "image": "c2_contact_phase_sync_current_en.png",
        "caption": "C2流程。在C1外层加入接触观测和有界相位修正，同时俯仰恢复预测仍与墙钟时间对齐。",
        "question": "C2要回答：如果实际触地/离地比计划早或晚，能否只调整相关腿的参考时间，同时保留C1的俯仰恢复能力？",
        "read": [
            "复用C1的FIR、AR、QP权重、G、恢复模式和SafetyLimiter；C2没有重新发明另一套俯仰控制器。",
            "从十步参考读取左右腿离地/触地计划事件，共20个计划事件，并保持事件顺序。",
            "Toe/heel接触传感器先合成为每条腿的接触状态，再检测观测事件{腿、类型、t_obs}。",
            "在0.22 s匹配窗口内，把观测事件与下一计划事件匹配；记录matched、skipped与unmatched，避免把错误事件强行配对。",
            "事件时间误差 epsilon_t=t_obs-t_s。章节中的实现公式为 Δphi^+=clip(Δphi+0.75 clip(epsilon_t,-0.04,0.04),-0.10,0.10)。",
            "内层clip限制单个事件最多贡献40 ms原始误差，0.75是更新增益，外层clip把总相位偏移限制在±100 ms。",
            "参考采用混合采样：所有关节先读取墙钟q_ref(t+0.030)，再只把下一事件腿替换为相位时间q_ref(t_phi+0.030)。",
            "俯仰误差仍是theta_meas-theta_baseline(t)，然后运行未改变的C1 FIR-QP修正，最后统一做关节可行投影和安全限制。",
        ],
        "symbols": [
            "t_s：计划接触事件时间；t_o/t_obs：观测接触事件时间；epsilon_t=t_o-t_s。",
            "Δphi：累积的参考时间偏移，不是关节角，也不是机器人几何相位角。",
            "t_phi=clip(t+Δphi,0,T)：事件腿使用的相位修正采样时间。",
            "leg-selective：只重定时下一事件相关腿；不是把全身参考整体向前或向后平移。",
        ],
        "conclusion": "C2把两个功能因子化组合：C1负责俯仰误差恢复，相位同步负责减少计划与观测接触事件的时间误差。",
        "boundary": "接触相位重置有文献先例，但这里的裁剪公式、0.75增益、40/100 ms边界和0.22 s窗口是项目设置。它不是新的普适混合系统稳定性定理。",
        "talk": "C2最关键的设计是‘只改下一事件腿的参考时间，不改墙钟俯仰目标’。所以相位同步不会把俯仰误差参考一起移动，C1仍能按原来的目标恢复。",
        "trace": "图源：figures/c2_contact_phase_sync_current_en.drawio/png。实现：mpc/multistep_controller.py；消融入口：mpc/18_validate_fddp_multistep_c2_final.py。",
        "theory": "理论依据：混合步态中的事件同步与相位重置；具体匹配、裁剪和腿选择逻辑属于项目实现。",
    },
    {
        "number": "图6.1",
        "title": "渐进式规划证据",
        "chapter": "第6章 实验结果",
        "image": "ch6_progressive_planning_evidence.png",
        "caption": "渐进式规划证据：单步与交替两步的前向CoM演化、摆脚净空、接触相位阴影及19 mm净空门。",
        "question": "这张图证明较短规划问题具有明确物理含义，并能作为十步连续扩展之前的开发与周期种子检查。",
        "read": [
            "实际排版是按列分任务、按行分变量：(a)单步CoM前向位置；(b)交替两步CoM前向位置；(c)单步左脚净空；(d)交替两步左右脚净空。",
            "横轴是时间s；上排纵轴是CoM前向位置mm；下排纵轴是摆脚相对初始高度的净空mm。",
            "黑色曲线表示CoM；蓝色表示左脚净空；橙色表示右脚净空；黑色虚线是19 mm接受门。",
            "背景阴影标记DS、LSS和RSS：双支撑、左脚摆动/右脚支撑、右脚摆动/左脚支撑。应该只在相应摆动相位检查摆脚峰值净空。",
            "单步用于验证摆动和落脚门；交替两步同时检查左右脚切换，并作为周期两步种子。",
        ],
        "symbols": [
            "CoM forward position：质心在论文前向坐标方向的位移，不是机身俯仰。",
            "Swing-foot clearance：摆脚接触框架相对初始地面的竖直抬升量。",
            "19 mm gate：项目冻结的摆脚净空接受门，不是所有机器人通用安全值。",
        ],
        "conclusion": "图支持单步和两步候选通过声明的物理净空与接触相位检查，并支持两步解作为后续周期连续化种子。",
        "boundary": "该图不展示FDDP一阶最优性或导数收敛。当前论文图注把(a)(b)说成单步、(c)(d)说成两步，与实际2x2排版不完全一致；建议改成“上排为单步/两步CoM，下排为对应净空”。",
        "talk": "先按列看：左列是单步，右列是两步；再按行看：上面是CoM，下面是摆脚净空。两条净空曲线都越过19 mm门，并且只在正确的单支撑区间抬脚，所以两步解可以继续扩展。",
        "trace": "生成：tools/generate_chapter6_publication_figures.py::plot_progressive_planning。数据：transfer/trajectories/fddp_single_step_left_swing_ref.npz 与 fddp_two_step_alternating_ref.npz。",
        "theory": "物理检查依据：质心前移、摆脚净空、接触序列和落脚连续性；不能替代求解器最优性证据。",
    },
    {
        "number": "图6.2",
        "title": "十步接触可行性审计",
        "chapter": "第6章 十步FDDP规划与接触",
        "image": "ch6_contact_feasibility.png",
        "caption": "十步轨迹在1300条激活接触记录上的平面合力可行性审计。",
        "question": "这张图检查被接受十步轨迹的每个支撑阶段，是否同时满足正法向力、摩擦、CoP裕度和脚底切向速度门。",
        "read": [
            "(a)合力法向力Fz，必须位于0 N以上；记录最小值9.912562 N。",
            "(b)摩擦利用率 rho_mu=sqrt(fx^2+fy^2)/(mu fz)，必须不超过0.90；记录最大值0.580708。",
            "(c)矢状面CoP裕度，必须至少2 mm；记录最小值2.391 mm。裕度越大表示压力中心离支撑边界越远。",
            "(d)脚底最大切向速度，门为0.1 m/s=100 mm/s；记录最大值为0，表示规划接触节点没有切向滑动。",
            "黑色曲线是节点上的统计量，橙色虚线是接受门，浅绿色区域是允许区域。每个面板右上角给出最不利记录。",
            "审计原始数据有1300条激活接触行；绘图代码为了形成时间曲线，按规划节点对同一节点的激活接触记录取均值。是否通过仍依据逐行审计，而不是只看图中均值。",
        ],
        "symbols": [
            "Fz>0：接触只能推地面，不能产生拉地面的法向力。",
            "rho_mu：摩擦需求相对可用Coulomb摩擦的比例；小于1通常表示在摩擦锥内，本项目使用更保守的0.90门。",
            "CoP margin：压力中心到足底支撑边界的最小距离。",
            "Tangential speed：接触脚底相对地面的切向速度，用于检查滑动。",
        ],
        "conclusion": "在论文声明的平面合力形式下，1300条激活接触记录零违规，十步候选通过接触门，可以进入MuJoCo回放。",
        "boundary": "这不是完整三维接触力矩锥证书：没有同等级验证偏航力矩等面外量；也没有证明存在满足真实舵机能力的关节力矩，因此不能称为硬件驱动可行性认证。",
        "talk": "四张子图分别对应支撑接触的四个必要检查。曲线都留在绿色允许区，最危险的是CoP裕度仍有2.391 mm。结论只到平面接触可接受，不延伸到完整3D或真实舵机。",
        "trace": "入口：fddp/05c_report_fddp_multistep_contact_feasibility.py。数据：results/logs/fddp_multistep_contact_feasibility.csv/yaml。绘图：plot_contact_feasibility。",
        "theory": "理论依据：单向接触、Coulomb摩擦、压力中心位于支撑区域内；驱动可行性是更强的另一层条件。",
    },
    {
        "number": "图6.3",
        "title": "十步C0完整MuJoCo回放",
        "chapter": "第6章 十步参考完整回放",
        "image": "ch6_nominal_replay_evidence.png",
        "caption": "完整10 s C0回放：机身俯仰、最大关节误差、FDDP/MuJoCo前向CoM和计划/观测接触状态。",
        "question": "规划与接触门通过以后，位置参考通过30 ms超前和执行器代理，能否在独立MuJoCo模型中完整执行10 s而不跌倒、腾空或发生接触顺序错误？",
        "read": [
            "(a)机身俯仰角，蓝线与±15 deg门比较；最大绝对值9.479 deg。",
            "(b)六关节中每一时刻的最大绝对跟踪误差，橙线与4 deg门比较；峰值2.203 deg。表中最大关节RMSE为1.124 deg，是另一种全时域统计量。",
            "(c)前向CoM变化：灰色虚线是FDDP参考，绿色实线是MuJoCo实际。两条线的接近程度表示位置回放保留了名义前移趋势。",
            "(d)接触状态阶梯线：左脚蓝色、右脚橙色；虚线是计划，实线是观测。纵向错开四条线只是为了可读，不代表接触力大小。",
            "20/20个计划事件按顺序匹配；完整运行没有跌倒、非预期腾空或持续穿透，最大穿透0.245 mm，最小高度比例0.9789。",
        ],
        "symbols": [
            "Maximum joint error：每个时间点先对六关节取绝对误差，再取其中最大者。",
            "Joint RMSE：每个关节在整个运行时域上的均方根误差；与瞬时峰值不能混用。",
            "Scheduled/observed contact：参考中计划的二值接触状态与MuJoCo传感器观测状态。",
        ],
        "conclusion": "C0在声明的执行器代理下完整回放十步参考并通过全部完整性门，说明离线轨迹具有仿真可执行性。",
        "boundary": "这是位置命令代理仿真，不是ST3215实物跟踪。图也不证明C0具有最强扰动恢复能力；它只建立后续控制器比较所需的名义基线。",
        "talk": "这一步把‘优化器说可行’变成‘独立MuJoCo确实能完整跑完’。四项证据分别看姿态、关节跟踪、前向运动和接触顺序，全部通过才接受C0。",
        "trace": "入口：simulation/03u_replay_fddp_multistep_mujoco.py。数据：results/logs/fddp_multistep_nominal_replay.csv/yaml。绘图：plot_nominal_replay。",
        "theory": "验证逻辑：规划可行性与闭环回放能力分层，不用规划器自身输出证明自己的物理执行结果。",
    },
    {
        "number": "图6.4",
        "title": "FIR符号保持验证",
        "chapter": "第6章 FIR预测结果",
        "image": "ch6_fir_holdout_validation.png",
        "caption": "向前与向后激励的符号保持FIR验证；实线为实测，虚线为由相反脉冲极性训练得到的预测，嵌入小图为残差。",
        "question": "FIR核是否只记住了用于辨识的那一个脉冲，还是能预测相反符号的局部响应？",
        "read": [
            "(a)向前扰动方向，(b)向后扰动方向；横轴时间s，纵轴根部俯仰偏差deg。",
            "蓝色对应被留出的负脉冲，橙色对应被留出的正脉冲；同色实线是MuJoCo实测，同色虚线是FIR预测。",
            "每个方向做两种划分：用正脉冲辨识、验证负脉冲；再用负脉冲辨识、验证正脉冲。",
            "右下角残差小图绘制 measured-predicted；越接近0表示预测越贴合。",
            "四个案例多步RMSE为0.0216-0.0224 deg，NRMSE为0.0546-0.0579，辨识到的输入-输出延迟为20 ms。",
        ],
        "symbols": [
            "Δtheta[k]≈sum h[i] alpha[k-i]：FIR卷积，过去M=23个alpha共同形成当前俯仰贡献。",
            "RMSE：预测与实测残差的均方根，单位deg。",
            "NRMSE：归一化RMSE，便于不同响应幅值之间比较；它不是百分比成功率。",
            "Held-out：该脉冲没有用于当前这条预测模型的拟合。",
        ],
        "conclusion": "FIR在被测试步态附近对相反脉冲极性具有很小的保持误差，足以支持其作为在线QP的局部命令响应模型。",
        "boundary": "每个方向只有一个正脉冲和一个负脉冲，且数据来自MuJoCo。不能据此声称全幅值、全步态相位或真实舵机上的全局模型有效性。",
        "talk": "我没有用训练曲线证明模型好，而是交换正负脉冲做保持验证。实线和虚线几乎重合，残差很小，但我只把结论限定在当前局部工作区间。",
        "trace": "辨识入口：mpc/19_identify_fddp_multistep_fir.py。数据：results/logs/e4_fir_holdout_predictions.csv 与 e4_fir_holdout_summary.yaml。绘图：plot_fir_holdout。",
        "theory": "理论依据：线性FIR卷积和保持数据验证；不能把局部输入-输出模型等同于完整多体动力学。",
    },
    {
        "number": "图6.5",
        "title": "C0/C1/Phase-only/C2完整时长消融",
        "chapter": "第6章 控制器消融与相位同步",
        "image": "ch6_controller_ablation.png",
        "caption": "同一十步参考上的四模式消融：俯仰误差、相位偏移、俯仰IAE和平均接触时序误差。",
        "question": "俯仰恢复和接触相位同步各自改善什么？两者组合后是否互相破坏？",
        "read": [
            "颜色固定：C0灰色、C1蓝色、Phase-only橙色、C2绿色。",
            "(a)完整10 s俯仰误差时间序列。C1与C2含FIR-QP修正；C0和Phase-only不含关节修正。",
            "(b)相位偏移时间序列。C0/C1保持0；Phase-only和C2根据接触事件产生有界偏移。",
            "(c)IAE柱状图：C0=3.7588、C1=3.3534、Phase-only=3.5378、C2=3.3897 deg s。数值越小表示累计俯仰误差越少。",
            "(d)平均绝对接触时序误差：C0=56.7、C1=56.6、Phase-only=36.7、C2=34.3 ms。相位同步负责主要改善。",
            "C2相对C1把时序误差降低39.4%；IAE从3.3534略变为3.3897，增幅约1.08%，仍低于预先声明的2%不退化门。",
            "最大关节修正：C0和Phase-only为0；C1和C2均为1.35 deg。",
        ],
        "symbols": [
            "C0：固定相位参考回放；C1：C0+FIR-QP；Phase-only：C0+相位同步；C2：C1+相位同步。",
            "IAE：integral |theta-theta_baseline| dt，既考虑误差大小也考虑持续时间。",
            "Mean absolute timing error：匹配接触事件的|t_obs-t_scheduled|平均值。",
            "2%不退化门：C2增加同步后，俯仰IAE相对C1不允许明显恶化。",
        ],
        "conclusion": "消融结果因子化支持：FIR-QP主要改善俯仰，接触相位同步主要改善事件时序，C2组合两者并通过俯仰不退化门。",
        "boundary": "这是一个固定十步参考和当前MuJoCo设置上的完整时长比较。不能推出所有接触条件下C2都优于C1，也不能替代硬件接触时序测试。",
        "talk": "这张图不要只说C2最好，要按功能讲。C1把俯仰IAE降下来，Phase-only把接触时序降下来，C2组合后时序最好，同时俯仰只比C1差1.08%，通过2%门。",
        "trace": "入口：mpc/18_validate_fddp_multistep_c2_final.py。数据：results/logs/fddp_multistep_controller_ablation.csv/yaml。绘图：plot_controller_ablation。",
        "theory": "实验依据：因子化消融，保持参考、模型、时长和限制相同，只切换功能模块。",
    },
    {
        "number": "图6.6",
        "title": "在线QP计时证据",
        "chapter": "第6章 控制器计时与软件安全",
        "image": "e6_controller_timing_distribution.png",
        "caption": "四种C1/C2配置的采样QP计时相对于20 ms控制截止期。",
        "question": "当前Python实现的小型QP是否在已记录更新中明显快于50 Hz控制周期？",
        "read": [
            "横轴是C1向前、C1向后、C2向前、C2向后四次控制器运行；每次记录61个QP更新，共244个。",
            "纵轴是QP求解时间us。每根柱实际表示该运行的P95求解时间，而不是完整概率分布。",
            "红色虚线是20 ms=20000 us控制周期。四根P95柱都在约0.75-1.08 ms，因纵轴包含20 ms门而视觉上很矮。",
            "P95最差为C1向后1082.5 us；全部样本最大值1507.7 us；截止期错过次数为0。",
            "表6.x同时给出P50、P95和最大值，解释时应配合表格，不应从柱高推断方差或尾部分布形状。",
        ],
        "symbols": [
            "P50：中位数，50%的样本不超过该值。",
            "P95：95%的样本不超过该值；仍有5%的样本可能更慢。",
            "Deadline miss：求解时间超过20000 us的样本。",
        ],
        "conclusion": "在当前机器、Python实现和244次采样更新中，QP求解相对20 ms周期具有明显计时余量。",
        "boundary": "文件名和图注使用“distribution”，但图实际是P95柱状汇总。它不是硬实时资格验证，也没有完成每模式10^5更新、P99/最坏执行时间、操作系统调度抖动或硬件通信计时。",
        "talk": "四种运行的P95都低于1.1 ms，而控制周期是20 ms，采样中没有超时。但我只称它为初步计时证据，不称为硬实时，因为样本量和运行环境还不够。",
        "trace": "生成：reporting/collect_controller_timing_safety.py。数据：results/logs/e6_controller_timing_safety.csv/yaml；论文表格给出四次运行的P50/P95/max。",
        "theory": "实时性结论需要比平均速度更严格的尾部、最坏情况、调度与I/O证据；本图只覆盖已采样QP计算时间。",
    },
    {
        "number": "图6.7",
        "title": "32场景C2鲁棒性包络",
        "chapter": "第6章 鲁棒性包络",
        "image": "ch6_robustness_envelope.png",
        "caption": "32个场景、每场景30个种子的峰值绝对俯仰误差包络与15 deg门。",
        "question": "冻结的C2控制器在已声明的推力、摩擦、坡度、台阶、延迟、执行器变化和噪声场景中，最严重俯仰响应距离15 deg门还有多少余量？",
        "read": [
            "纵轴列出32个鲁棒性场景；横轴是每个场景30个种子的峰值绝对俯仰误差deg。",
            "每条灰色水平线从该场景30个种子的最小值延伸到最大值；圆点是均值。",
            "实心圆是预先标记的关键场景，空心圆是非关键场景；颜色区分nominal、push、friction、slope、step、delay、actuator与noise等类别。",
            "竖向黑色虚线是15 deg接受门。所有水平范围都在门的左侧。",
            "32x30=960次完整试验全部接受；观测严重程度总范围约1.7022-13.7048 deg。",
            "每个30/30单元的Wilson 95%成功率下界仍只有0.8865，说明有限样本全通过不等于真实成功概率为100%。",
        ],
        "symbols": [
            "Peak absolute pitch error：一次10 s试验内|俯仰误差|的最大值。",
            "Min/mean/max range：同一场景30个确定性种子的最小、平均和最大严重程度。",
            "Wilson lower bound：对二项成功率的有限样本置信下界；30/30时仍低于0.90。",
            "Critical cell：试验设计中预先认定必须重点通过的场景，不是看完结果后才选择。",
        ],
        "conclusion": "冻结C2在论文声明的32场景采样包络中完成960/960接受，并且所有记录的峰值俯仰都低于15 deg门。",
        "boundary": "结论只覆盖这些参数范围和30个种子。它不证明任意推力、任意地形、任意摩擦或真实硬件下必然稳定，也不能把960/960写成普适100%成功率。",
        "talk": "我不再画32根都等于100%的成功率柱，而是画每个场景的严重程度范围。最危险场景接近13.7 deg但仍低于15 deg；同时Wilson下界提醒我们样本仍有限。",
        "trace": "入口：mpc/21_run_fddp_multistep_robustness_matrix.py。数据：results/logs/fddp_multistep_robustness_trials.csv、aggregate.csv与summary.yaml。绘图：plot_robustness_envelope。",
        "theory": "鲁棒性证据必须同时报告场景范围、种子数、严重程度和统计不确定性，而不仅报告通过率。",
    },
    {
        "number": "图6.8",
        "title": "±4 N有界推力恢复分布",
        "chapter": "第6章 有界推力恢复",
        "image": "ch6_bounded_push_recovery.png",
        "caption": "实际执行的fir_fast候选在+4 N和-4 N推力下的30种子恢复分布与15 deg门。",
        "question": "预先排序的恢复候选中，第一个fir_fast是否已经同时通过正反两个方向，从而触发顺序停止？",
        "read": [
            "(a)向前+4 N，(b)向后-4 N。每个彩色点是一颗种子的峰值绝对俯仰误差，箱线图汇总中位数和四分位范围。",
            "黑色虚线是15 deg门；+4 N最大值9.0417 deg，-4 N最大值13.7454 deg，两个方向均30/30接受。",
            "运行中记录max|alpha|=5 deg，等于QP修正幅值边界；max|Δq|=1.35 deg，与G中最大系数0.27乘5 deg一致。",
            "图中saturation fraction=0来自记录字段。由于alpha达到5 deg边界，解释时应把它表述为“记录的特定饱和/限幅计数为0”，不要泛化成所有约束从未激活。",
            "顺序候选规则：fir_fast先运行；它双向通过后，后续候选标记为not_run_earlier_stage_accepted。",
            "落脚点重规划没有执行，因为那会改变名义参考并需要重新生成规划与接触证据。",
        ],
        "symbols": [
            "Push force：仿真中施加的有界外力，不代表现实中某一种具体碰撞冲量。",
            "Boxplot：箱体表示25%-75%分位区间，中线是中位数，须结合全部散点而不是只看箱体。",
            "alpha：QP标量修正；Δq=G alpha：实际六关节位置修正。",
            "Sequential stopping：前一个候选满足预设通过规则后停止，不再浪费计算运行后续候选。",
        ],
        "conclusion": "fir_fast在已执行的±4 N、各30种子试验中双向通过，因此按预设顺序规则被接受。",
        "boundary": "这不是三种候选的完整横向比较，因为后续候选未运行；也不是落脚点重规划实验。图只支持当前推力定义、时长、方向和模型内的恢复结论。",
        "talk": "这张图要讲清楚‘只运行了一个候选’。fir_fast在正反4 N各30次都通过，所以按规则停止；我不能拿这张图比较三个候选，更不能说做过落脚重规划。",
        "trace": "入口：mpc/19_tune_fddp_multistep_push_recovery.py。数据：results/logs/fddp_multistep_recovery_sweep.csv 与 recovery_selection.yaml。绘图：plot_bounded_push_recovery。",
        "theory": "顺序试验设计可以减少不必要运行，但必须透明报告未运行候选，避免把“未运行”写成“表现更差”。",
    },
]


def page_footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D9E0E5"))
    canvas.setLineWidth(0.4)
    canvas.line(LEFT_MARGIN, 11 * mm, PAGE_WIDTH - RIGHT_MARGIN, 11 * mm)
    canvas.setFont(FONT_NAME, 7.5)
    canvas.setFillColor(MID_GREY)
    canvas.drawString(LEFT_MARGIN, 6.5 * mm, "双足机器人论文全部图详解")
    canvas.drawRightString(
        PAGE_WIDTH - RIGHT_MARGIN,
        6.5 * mm,
        f"第 {document.page} 页",
    )
    canvas.restoreState()


def build_story(style_map, generated_images: dict[str, Path]):
    story = []
    story.extend(
        [
            Spacer(1, 28 * mm),
            paragraph("双足机器人论文全部图详解", style_map["title"]),
            paragraph(
                "覆盖论文15张正式图：图面元素、公式符号、证据结论、限制边界、答辩讲法与代码追溯",
                style_map["subtitle"],
            ),
            Spacer(1, 13 * mm),
            info_box(
                "这份文件怎么用",
                "它不是把论文图注重复一遍，而是把每张图翻译成可理解、可答辩、可追溯的语言。建议先看“一句话目的”和“逐项读取”，再看“能支持的结论”和“不能过度声称什么”。",
                style_map["box"],
                LIGHT_BLUE,
                BLUE,
            ),
            Spacer(1, 8 * mm),
            info_box(
                "全篇统一证据边界",
                "当前论文支持离线FDDP规划、位置命令执行器代理下的MuJoCo完整回放、FIR-QP-MPC与接触相位同步的仿真消融、采样鲁棒性和初步计时。它不支持真实ST3215跟踪、实物机器人行走或硬实时资格结论。",
                style_map["box"],
                LIGHT_RED,
                RED,
            ),
            Spacer(1, 10 * mm),
            paragraph(
                "版本依据：Overleaf main 提交 ab646fa；生成日期：2026-07-29。两张TikZ图根据当前论文源码重绘，其余13张直接复用论文正式图文件。",
                style_map["small"],
            ),
            PageBreak(),
            paragraph("目录与主线", style_map["h1"]),
            paragraph(
                "15张图可以按四段主线理解：研究定位 -> 系统架构 -> 规划与控制方法 -> 仿真证据。",
                style_map["body"],
            ),
        ]
    )
    toc_rows = [["图号", "主题", "核心问题"]]
    for figure in FIGURES:
        toc_rows.append([figure["number"], figure["title"], figure["question"]])
    toc = Table(
        [
            [
                paragraph(str(cell), style_map["toc"])
                for cell in row
            ]
            for row in toc_rows
        ],
        colWidths=[18 * mm, 48 * mm, CONTENT_WIDTH - 66 * mm],
        repeatRows=1,
    )
    toc.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD_NAME),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CDD5DB")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(toc)
    story.extend(
        [
            Spacer(1, 7 * mm),
            info_box(
                "一条可以从头讲到尾的答辩主线",
                "图2.1说明研究位置；图3.1说明系统如何闭环；图4.1说明离线FDDP怎样生成被接受参考；图4.2-4.5说明C0、MPC、C1、C2逐层增加什么；图6.1-6.8依次回答轨迹是否物理合理、接触是否可行、能否完整回放、预测模型是否可信、控制层各自是否有效、在线计算是否够快、采样鲁棒性如何以及推力恢复是否通过。",
                style_map["box"],
                LIGHT_GREEN,
                GREEN,
            )
        ]
    )

    for figure in FIGURES:
        image_path = (
            generated_images[figure["image"]]
            if figure.get("reconstructed")
            else FIGURE_ROOT / figure["image"]
        )
        story.extend(
            [
                PageBreak(),
                paragraph(
                    f'{figure["number"]}  {figure["title"]}',
                    style_map["h1"],
                ),
                paragraph(f'论文位置：{figure["chapter"]}', style_map["small"]),
                Spacer(1, 3 * mm),
                scaled_image(image_path),
                paragraph(f'论文图注：{figure["caption"]}', style_map["caption"]),
                info_box(
                    "这张图要回答什么",
                    figure["question"],
                    style_map["box"],
                    LIGHT_BLUE,
                    BLUE,
                ),
                paragraph("怎样逐项读图", style_map["h2"]),
                bullets(figure["read"], style_map["bullet"]),
                paragraph("公式与符号", style_map["h2"]),
                bullets(figure["symbols"], style_map["bullet"]),
                info_box(
                    "这张图能够支持的结论",
                    figure["conclusion"],
                    style_map["box"],
                    LIGHT_GREEN,
                    GREEN,
                ),
                Spacer(1, 3 * mm),
                info_box(
                    "不能过度声称什么",
                    figure["boundary"],
                    style_map["box"],
                    LIGHT_RED,
                    RED,
                ),
                paragraph("组会/答辩可以这样讲", style_map["h2"]),
                paragraph(figure["talk"], style_map["body"]),
                paragraph("代码、数据与理论追溯", style_map["h2"]),
                paragraph(figure["trace"], style_map["small"]),
                paragraph(figure["theory"], style_map["small"]),
            ]
        )

    story.extend(
        [
            PageBreak(),
            paragraph("全篇符号速查", style_map["h1"]),
            info_box(
                "规划与执行的最关键区别",
                "tau_ref属于FDDP规划诊断；q_ref是离线名义位置参考；Delta q是在线修正；q_cmd=q_ref+Delta q才是执行器代理和未来硬件运行时真正接收的位置命令。",
                style_map["box"],
                LIGHT_ORANGE,
                ORANGE,
            ),
        ]
    )
    symbol_rows = [
        ("x_k=(q_k,v_k)", "FDDP节点状态：构型与速度"),
        ("u_k=tau_k", "FDDP规划控制：六关节力矩，仅用于规划模型"),
        ("g_k", "多重射击相邻节点的动力学连续性缺陷"),
        ("k_k, K_k", "DDP/FDDP反向传递得到的局部前馈方向与反馈矩阵"),
        ("q_ref", "被接受的离线关节位置参考"),
        ("theta_baseline", "C0名义MuJoCo俯仰基线"),
        ("e_theta", "相对名义基线的俯仰误差"),
        ("h_i", "FIR脉冲响应系数"),
        ("a_j", "AR自然响应系数"),
        ("A", "未来预测时域内的alpha序列"),
        ("S", "由FIR核构成的卷积/动态矩阵"),
        ("G", "标量alpha到六关节位置修正的映射"),
        ("Delta phi", "由接触事件更新时间误差得到的有界参考时间偏移"),
        ("IAE", "绝对误差随时间积分，兼顾大小与持续时间"),
        ("Wilson下界", "有限二项样本成功率的不确定性下界"),
    ]
    symbol_table = Table(
        [
            [paragraph("符号", style_map["toc"]), paragraph("项目中的含义", style_map["toc"])]
        ]
        + [
            [paragraph(symbol, style_map["toc"]), paragraph(description, style_map["toc"])]
            for symbol, description in symbol_rows
        ],
        colWidths=[45 * mm, CONTENT_WIDTH - 45 * mm],
        repeatRows=1,
    )
    symbol_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CDD5DB")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(symbol_table)
    story.extend(
        [
            PageBreak(),
            paragraph("答辩时最容易说错的八件事", style_map["h1"]),
            bullets(
                [
                    "不要说七阶段同伦就是七次FDDP迭代；它是七个逐步加权、彼此热启动的优化问题。",
                    "不要说tau_ref被发送给舵机；当前硬件方向是q_cmd位置命令。",
                    "不要把C0说成完全无控制；它没有高层修正，但仍有位置执行器代理和SafetyLimiter。",
                    "不要把C1称为全身非线性MPC；它是局部FIR/AR预测加标量QP。",
                    "不要说C2改变了全身时间或俯仰参考；它只重定时下一事件腿，俯仰基线仍按墙钟时间。",
                    "不要把接触审计说成完整3D和驱动可行性证明；当前主要结论是平面合力审计。",
                    "不要把960/960说成对任意扰动100%稳定；Wilson下界和场景范围必须同时报告。",
                    "不要把QP计时图说成硬实时认证；它只有244个采样更新和P95/最大值证据。",
                ],
                style_map["bullet"],
            ),
            Spacer(1, 7 * mm),
            info_box(
                "一句话总总结",
                "这15张图共同支持的最强结论是：项目已经形成离线FDDP名义规划、位置命令代理下MuJoCo回放、FIR-QP俯仰修正、接触事件相位同步、控制器消融、采样鲁棒性与初步计时的完整仿真证据链；真实硬件仍是下一阶段。",
                style_map["box"],
                LIGHT_GREEN,
                GREEN,
            ),
        ]
    )
    return story


def validate_pdf(path: Path) -> None:
    reader = PdfReader(str(path))
    if len(reader.pages) < 20:
        raise RuntimeError(f"unexpectedly short PDF: {len(reader.pages)} pages")
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    required = [
        "图2.1",
        "图3.1",
        "图4.1",
        "图4.5",
        "图6.1",
        "图6.8",
        "不能过度声称什么",
        "全篇符号速查",
    ]
    missing = [value for value in required if value not in text]
    if missing:
        raise RuntimeError(f"PDF text validation failed: {missing}")


def main() -> int:
    register_fonts()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    positioning = TMP_DIR / "reconstructed_positioning.png"
    architecture = TMP_DIR / "reconstructed_architecture.png"
    reconstruct_positioning(positioning)
    reconstruct_architecture(architecture)
    generated_images = {
        "reconstructed_positioning.png": positioning,
        "reconstructed_architecture.png": architecture,
    }

    document = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=LEFT_MARGIN,
        rightMargin=RIGHT_MARGIN,
        topMargin=TOP_MARGIN,
        bottomMargin=BOTTOM_MARGIN,
        title="双足机器人论文全部图详解",
        author="Codex",
        subject="论文15张正式图的详细中文解释、答辩讲法与证据边界",
    )
    style_map = styles()
    document.build(
        build_story(style_map, generated_images),
        onFirstPage=page_footer,
        onLaterPages=page_footer,
    )
    validate_pdf(OUTPUT_PDF)
    positioning.unlink(missing_ok=True)
    architecture.unlink(missing_ok=True)
    print(OUTPUT_PDF)
    print(f"pages={len(PdfReader(str(OUTPUT_PDF)).pages)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
