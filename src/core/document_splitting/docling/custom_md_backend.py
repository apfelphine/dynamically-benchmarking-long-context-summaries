import re
from copy import deepcopy
from html import unescape
from pathlib import Path
from typing import Optional, Union, cast

import marko
import marko.element
import marko.inline
from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    ListItem,
    NodeItem,
    TextItem,
)
from docling_core.types.doc.document import Formatting
from pydantic import AnyUrl, TypeAdapter

from docling.backend.md_backend import (
    MarkdownDocumentBackend as DoclingMdBackend,
    _CreationPayload,  # noqa
    _log,  # noqa
    _HeadingCreationPayload,  # noqa
    _ListItemCreationPayload,  # noqa
    _START_MARKER,  # noqa
    _STOP_MARKER,  # noqa
)


class MarkdownDocumentBackend(DoclingMdBackend):
    def _iterate_elements(
        self,
        *,
        element: marko.element.Element,
        depth: int,
        doc: DoclingDocument,
        visited: set[marko.element.Element],
        creation_stack: list[
            _CreationPayload
        ],  # stack for lazy item creation triggered deep in marko's AST (on RawText)
        list_ordered_flag_by_ref: dict[str, bool],
        list_last_item_by_ref: dict[str, ListItem],
        parent_item: Optional[NodeItem] = None,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
        last_element=True,
    ):
        if element in visited:
            return

        # Iterates over all elements in the AST
        # Check for different element types and process relevant details
        if (
            isinstance(element, marko.block.Heading)
            or isinstance(element, marko.block.SetextHeading)
        ) and len(element.children) > 0:
            self._close_table(doc)
            _log.debug(
                f" - Heading level {element.level}, content: {element.children[0].children}"  # type: ignore
            )

            if len(element.children) > 1:  # inline group will be created further down
                parent_item = self._create_heading_item(
                    doc=doc,
                    parent_item=parent_item,
                    text="",
                    level=element.level,
                    formatting=formatting,
                    hyperlink=hyperlink,
                )
            else:
                creation_stack.append(_HeadingCreationPayload(level=element.level))

        elif isinstance(element, marko.block.List):
            has_non_empty_list_items = False
            for child in element.children:
                if isinstance(child, marko.block.ListItem) and len(child.children) > 0:
                    has_non_empty_list_items = True
                    break

            self._close_table(doc)
            _log.debug(f" - List {'ordered' if element.ordered else 'unordered'}")
            if has_non_empty_list_items:
                parent_item = doc.add_list_group(name="list", parent=parent_item)
                list_ordered_flag_by_ref[parent_item.self_ref] = element.ordered

        elif (
            isinstance(element, marko.block.ListItem)
            and len(element.children) > 0
            and isinstance((child := element.children[0]), marko.block.Paragraph)
            and len(child.children) > 0
        ):
            self._close_table(doc)
            _log.debug(" - List item")

            enumerated = (
                list_ordered_flag_by_ref.get(parent_item.self_ref, False)
                if parent_item
                else False
            )
            non_list_children: list[marko.element.Element] = [
                item
                for item in child.children
                if not isinstance(item, marko.block.ListItem)
            ]
            if len(non_list_children) > 1:  # inline group will be created further down
                parent_ref: Optional[str] = (
                    parent_item.self_ref if parent_item else None
                )
                parent_item = self._create_list_item(
                    doc=doc,
                    parent_item=parent_item,
                    text="",
                    enumerated=enumerated,
                    formatting=formatting,
                    hyperlink=hyperlink,
                )
                if parent_ref:
                    list_last_item_by_ref[parent_ref] = cast(ListItem, parent_item)
            else:
                creation_stack.append(_ListItemCreationPayload(enumerated=enumerated))

        # ############# START OF CUSTOM LOGIC #################
        elif isinstance(element, marko.block.Paragraph):
            is_custom_heading = False
            heading_text = ""
            remaining_children_start_idx = 0
            # Helper to extract clean text from Bold node

            # Detection Custom Header
            # Check if first paragraph child is bold
            if len(element.children) > 0 and isinstance(
                element.children[0], marko.inline.StrongEmphasis
            ):
                candidate_text = self._get_clean_bold_text(element.children[0])
                if candidate_text:

                    # If paragraph ONLY contains bold text, it is a custom header (as long as this isn't the LAST element)
                    if len(element.children) == 1:
                        if not last_element:
                            is_custom_heading = True
                            heading_text = candidate_text
                            remaining_children_start_idx = 1

                    # If there is a new line after the bold text, it is a custom header
                    else:
                        second_child = element.children[1]

                        # Explicit LineBreak
                        if isinstance(second_child, marko.inline.LineBreak):
                            is_custom_heading = True
                            heading_text = candidate_text
                            remaining_children_start_idx = 2

                        # RawText starting with a newline
                        elif isinstance(
                            second_child, (marko.inline.RawText, marko.inline.Literal)
                        ):
                            text_content = second_child.children

                            if isinstance(text_content, str):
                                if text_content.startswith("\n") or text_content.lstrip(
                                    " "
                                ).startswith("\n"):
                                    is_custom_heading = True
                                    heading_text = candidate_text

                                    cleaned = text_content.lstrip(" ").lstrip("\n")
                                    element.children[1].children = cleaned  # noqa
                                    remaining_children_start_idx = 1

            if is_custom_heading:
                self._close_table(doc)
                _log.debug(f" - Custom Split Heading: {heading_text}")

                # Create the Heading
                self._create_heading_item(
                    doc=doc,
                    parent_item=parent_item,
                    text=heading_text,
                    level=4,
                    formatting=formatting,
                    hyperlink=hyperlink,
                )

                # Process the REST of the paragraph
                remaining_children = element.children[remaining_children_start_idx:]
                if len(remaining_children) > 0:
                    if len(remaining_children) > 1:
                        # Mimic inline group
                        target_parent = doc.add_inline_group(parent=parent_item)
                    else:
                        # Single item can usually attach directly to parent without breaking flow
                        target_parent = parent_item

                    for idx, child in enumerate(remaining_children):
                        self._iterate_elements(
                            element=child,
                            depth=depth + 1,
                            doc=doc,
                            visited=visited,
                            creation_stack=creation_stack,
                            list_ordered_flag_by_ref=list_ordered_flag_by_ref,
                            list_last_item_by_ref=list_last_item_by_ref,
                            parent_item=target_parent,
                            formatting=formatting,
                            hyperlink=hyperlink,
                            last_element=idx == len(remaining_children) - 1,
                        )
                return
        # ############# END OF CUSTOM LOGIC #################

        elif isinstance(element, marko.inline.Image):
            self._close_table(doc)
            _log.debug(f" - Image with alt: {element.title}, url: {element.dest}")

            fig_caption: Optional[TextItem] = None
            if element.title is not None and element.title != "":
                title = unescape(element.title)
                fig_caption = doc.add_text(
                    label=DocItemLabel.CAPTION,
                    text=title,
                    formatting=formatting,
                    hyperlink=hyperlink,
                )

            doc.add_picture(parent=parent_item, caption=fig_caption)

        elif isinstance(element, marko.inline.Emphasis):
            _log.debug(f" - Emphasis: {element.children}")
            formatting = deepcopy(formatting) if formatting else Formatting()
            formatting.italic = True

        elif isinstance(element, marko.inline.StrongEmphasis):
            _log.debug(f" - StrongEmphasis: {element.children}")
            formatting = deepcopy(formatting) if formatting else Formatting()
            formatting.bold = True

        elif isinstance(element, marko.inline.Link):
            _log.debug(f" - Link: {element.children}")
            hyperlink = TypeAdapter(Optional[Union[AnyUrl, Path]]).validate_python(
                element.dest
            )

        elif isinstance(element, (marko.inline.RawText, marko.inline.Literal)):
            _log.debug(f" - RawText/Literal: {element.children}")
            snippet_text = (
                element.children.strip() if isinstance(element.children, str) else ""
            )
            snippet_text = unescape(snippet_text)

            # ############# START OF CUSTOM LOGIC #################
            pattern = (
                r"\\."  # 1. Escaped chars (ignore)
                r"|(\$\$(?:.|\n)*?\$\$)"  # 2. Double Dollar (allows newlines, ignore)
                r"|(\$[^$\n]*\$)"  # 3. Single Dollar (NO newlines, ignore)
                r"|(\|)"  # 4. A Pipe that isn't in math mode (Signal for a table)
            )
            found_pipes = 0
            for match in re.finditer(pattern, snippet_text):
                if match.group(3) == "|":
                    # Count the pipes cus sometimes the $$ of math are not in this element
                    # A table start gotta have more than one pipe (e.g. | Hello |)
                    found_pipes += 1
            if found_pipes > 1 or self.in_table:
                # ############# END OF CUSTOM LOGIC ##################

                # most likely part of the Markdown table
                self.in_table = True
                if len(self.md_table_buffer) > 0:
                    self.md_table_buffer[len(self.md_table_buffer) - 1] += snippet_text
                else:
                    self.md_table_buffer.append(snippet_text)
            elif snippet_text:
                self._close_table(doc)

                if creation_stack:
                    while len(creation_stack) > 0:
                        to_create = creation_stack.pop()
                        if isinstance(to_create, _ListItemCreationPayload):
                            enumerated = (
                                list_ordered_flag_by_ref.get(
                                    parent_item.self_ref, False
                                )
                                if parent_item
                                else False
                            )
                            parent_ref = parent_item.self_ref if parent_item else None
                            parent_item = self._create_list_item(
                                doc=doc,
                                parent_item=parent_item,
                                text=snippet_text,
                                enumerated=enumerated,
                                formatting=formatting,
                                hyperlink=hyperlink,
                            )
                            if parent_ref:
                                list_last_item_by_ref[parent_ref] = cast(
                                    ListItem, parent_item
                                )

                        elif isinstance(to_create, _HeadingCreationPayload):
                            # not keeping as parent_item as logic for correctly tracking
                            # that not implemented yet (section components not captured
                            # as heading children in marko)
                            self._create_heading_item(
                                doc=doc,
                                parent_item=parent_item,
                                text=snippet_text,
                                level=to_create.level,
                                formatting=formatting,
                                hyperlink=hyperlink,
                            )
                else:
                    doc.add_text(
                        label=DocItemLabel.TEXT,
                        parent=parent_item,
                        text=snippet_text,
                        formatting=formatting,
                        hyperlink=hyperlink,
                    )

        elif isinstance(element, marko.inline.CodeSpan):
            self._close_table(doc)
            _log.debug(f" - Code Span: {element.children}")
            snippet_text = str(element.children).strip()
            doc.add_code(
                parent=parent_item,
                text=snippet_text,
                formatting=formatting,
                hyperlink=hyperlink,
            )

        elif (
            isinstance(element, (marko.block.CodeBlock, marko.block.FencedCode))
            and len(element.children) > 0
            and isinstance((child := element.children[0]), marko.inline.RawText)
            and len(snippet_text := (child.children.strip())) > 0
        ):
            self._close_table(doc)
            _log.debug(f" - Code Block: {element.children}")
            doc.add_code(
                parent=parent_item,
                text=snippet_text,
                formatting=formatting,
                hyperlink=hyperlink,
            )

        elif isinstance(element, marko.inline.LineBreak):
            if self.in_table:
                _log.debug("Line break in a table")
                self.md_table_buffer.append("")

        elif isinstance(element, marko.block.HTMLBlock):
            self._html_blocks += 1
            self._close_table(doc)
            _log.debug(f"HTML Block: {element}")
            if (
                len(element.body) > 0
            ):  # If Marko doesn't return any content for HTML block, skip it
                html_block = element.body.strip()

                # wrap in markers to enable post-processing in convert()
                text_to_add = f"{_START_MARKER}{html_block}{_STOP_MARKER}"
                doc.add_code(
                    parent=parent_item,
                    text=text_to_add,
                    formatting=formatting,
                    hyperlink=hyperlink,
                )
        else:
            if not isinstance(element, str):
                self._close_table(doc)
                _log.debug(f"Some other element: {element}")

        if (
            isinstance(element, (marko.block.Paragraph, marko.block.Heading))
            and len(element.children) > 1
        ):
            parent_item = doc.add_inline_group(parent=parent_item)

        processed_block_types = (
            marko.block.CodeBlock,
            marko.block.FencedCode,
            marko.inline.RawText,
        )

        # Iterate through the element's children (if any)
        if hasattr(element, "children") and not isinstance(
            element, processed_block_types
        ):
            for idx, child in enumerate(element.children):
                if (
                    isinstance(element, marko.block.ListItem)
                    and isinstance(child, marko.block.List)
                    and parent_item
                    and list_last_item_by_ref.get(parent_item.self_ref, None)
                ):
                    _log.debug(
                        f"walking into new List hanging from item of parent list {parent_item.self_ref}"
                    )
                    parent_item = list_last_item_by_ref[parent_item.self_ref]

                self._iterate_elements(
                    element=child,
                    depth=depth + 1,
                    doc=doc,
                    visited=visited,
                    creation_stack=creation_stack,
                    list_ordered_flag_by_ref=list_ordered_flag_by_ref,
                    list_last_item_by_ref=list_last_item_by_ref,
                    parent_item=parent_item,
                    formatting=formatting,
                    hyperlink=hyperlink,
                    last_element=idx == len(element.children) - 1,
                )

    @staticmethod
    def _get_clean_bold_text(node):
        parts = []

        if hasattr(node, "children"):
            for fc in node.children:
                if isinstance(fc, (marko.inline.RawText, marko.inline.Literal)):
                    parts.append(fc.children)
                elif isinstance(fc, str):
                    parts.append(fc)

        return unescape("".join(parts).strip()).replace("\n", " ")
