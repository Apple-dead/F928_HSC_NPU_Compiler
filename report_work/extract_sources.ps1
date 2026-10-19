$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$out = Join-Path $PSScriptRoot 'source_extract'
New-Item -ItemType Directory -Path $out -Force | Out-Null
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $names = @('附件1：中期检查格式及要求.doc','附件2：参考文献书写格式.doc','蒋一平-开题报告.docx','目录示例.docx','蒋一平-文献综述.docx')
    foreach ($name in $names) {
        $path = Join-Path $root $name
        $doc = $word.Documents.Open($path, $false, $true)
        $stem = [System.IO.Path]::GetFileNameWithoutExtension($name)
        [System.IO.File]::WriteAllText((Join-Path $out ($stem+'.txt')), $doc.Content.Text, [System.Text.Encoding]::UTF8)
        $info = [ordered]@{name=$name; pages=$doc.ComputeStatistics(2); width=$doc.PageSetup.PageWidth; height=$doc.PageSetup.PageHeight; top=$doc.PageSetup.TopMargin; bottom=$doc.PageSetup.BottomMargin; left=$doc.PageSetup.LeftMargin; right=$doc.PageSetup.RightMargin; paragraphs=@()}
        foreach($p in $doc.Paragraphs){
            $t=$p.Range.Text.Trim()
            if($t.Length -gt 0){
                $info.paragraphs += [ordered]@{text=$t; style=$p.Style.NameLocal; font=$p.Range.Font.NameFarEast; latin=$p.Range.Font.NameAscii; size=$p.Range.Font.Size; bold=$p.Range.Font.Bold; align=$p.Format.Alignment; first=$p.Format.FirstLineIndent; line=$p.Format.LineSpacing; lineRule=$p.Format.LineSpacingRule; before=$p.Format.SpaceBefore; after=$p.Format.SpaceAfter}
            }
        }
        $info | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 (Join-Path $out ($stem+'.format.json'))
        if($name.EndsWith('.doc')){ $doc.SaveAs2((Join-Path $out ($stem+'.docx')),16) }
        $doc.Close(0)
        Write-Output ('Extracted: '+$name)
    }
} finally { $word.Quit(); [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) }
