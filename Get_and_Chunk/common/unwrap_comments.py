import re

# Read the file
with open('merged_output_no_line_breaks.md', 'r', encoding='utf-8') as f:
    content = f.read()

def unwrap_comments(text):
    """
    Unwrap multi-line // comments by:
    1. Finding comment blocks (consecutive lines starting with //)
    2. Keeping the first line as-is
    3. Removing the // and joining continuation lines to the first line
    """
    lines = text.split('\n')
    result_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this line starts a comment block
        if line.strip().startswith('//') and not line.strip().startswith('///'):
            # This is the start of a comment block
            comment_lines = [line]
            i += 1
            
            # Collect all continuation comment lines
            while i < len(lines):
                next_line = lines[i]
                if next_line.strip().startswith('//') and not next_line.strip().startswith('///'):
                    comment_lines.append(next_line)
                    i += 1
                else:
                    break
            
            # Process the comment block
            if len(comment_lines) > 1:
                # Keep the first line as-is
                first_line = comment_lines[0]
                
                # Extract text content from continuation lines (remove // and leading whitespace)
                continuation_texts = []
                for comment_line in comment_lines[1:]:
                    # Remove the // and any leading/trailing whitespace
                    text_part = comment_line.strip()
                    if text_part.startswith('//'):
                        text_part = text_part[2:].strip()
                        if text_part:  # Only add non-empty text
                            continuation_texts.append(text_part)
                
                # Combine first line with continuation text
                if continuation_texts:
                    # Extract the text part from the first line (everything after //)
                    first_text = first_line.strip()
                    if first_text.startswith('//'):
                        first_text = first_text[2:].strip()
                    
                    # Get the indentation from the first line
                    first_line_indent = len(first_line) - len(first_line.lstrip())
                    indent = ' ' * first_line_indent
                    
                    # Combine all text parts
                    combined_text = first_text
                    if combined_text and continuation_texts:
                        combined_text += ' ' + ' '.join(continuation_texts)
                    elif continuation_texts:
                        combined_text = ' '.join(continuation_texts)
                    
                    # Reconstruct the comment line
                    result_lines.append(f"{indent}//{' ' + combined_text if combined_text else ''}")
                else:
                    result_lines.append(first_line)
            else:
                # Single line comment, keep as-is
                result_lines.append(comment_lines[0])
        else:
            # Not a comment line, keep as-is
            result_lines.append(line)
            i += 1
    
    return '\n'.join(result_lines)

# Process the content
unwrapped_content = unwrap_comments(content)

# Write the result
with open('merged_output_unwrapped_comments.md', 'w', encoding='utf-8') as f:
    f.write(unwrapped_content)

print("Comment unwrapping complete! Output saved to merged_output_unwrapped_comments.md")
print("Multi-line // comments have been unwrapped into single flowing lines.")