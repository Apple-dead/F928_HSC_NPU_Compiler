$ErrorActionPreference='Stop'
$root=(Get-Location).Path
$path=Join-Path $root '论文中期材料\蒋一平-硕士论文中期检查报告（研究计划与示意图修订稿）.docx'
$qa=Join-Path $root 'report_work\plan_figures_render'
New-Item -ItemType Directory -Path $qa -Force | Out-Null
$app=New-Object -ComObject kwps.Application
$app.Visible=$false
$app.DisplayAlerts=0
$d=$null
try {
 $d=$app.Documents.Open($path,$false,$false)
 $d.Repaginate()
 foreach($toc in $d.TablesOfContents){$toc.Update()}
 $d.Fields.Update() | Out-Null
 $d.Repaginate()
 foreach($toc in $d.TablesOfContents){$toc.UpdatePageNumbers()}
 $d.Save()
 $d.ExportAsFixedFormat((Join-Path $qa 'midterm_revision.pdf'),17)
 Write-Output ('pages='+$d.ComputeStatistics(2))
} finally {
 if($null -ne $d){$d.Close(0)}
 $app.Quit()
}



