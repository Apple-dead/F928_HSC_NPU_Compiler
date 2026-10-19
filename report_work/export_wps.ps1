$ErrorActionPreference='Stop'
$root=(Get-Location).Path
$qa=Join-Path $root 'report_work\render'
New-Item -ItemType Directory -Path $qa -Force | Out-Null
$app=New-Object -ComObject kwps.Application
$app.Visible=$false
$app.DisplayAlerts=0
try {
 foreach($f in Get-ChildItem -LiteralPath (Join-Path $root '论文中期材料') -Filter '*.docx') {
  $d=$null
  try {
   $d=$app.Documents.Open($f.FullName,$false,$false)
   $d.Repaginate()
   foreach($toc in $d.TablesOfContents){$toc.Update()}
   $d.Fields.Update() | Out-Null
   $d.Repaginate()
   foreach($toc in $d.TablesOfContents){$toc.UpdatePageNumbers()}
   $d.Save()
   $pdf=Join-Path $qa ($f.BaseName+'.pdf')
   $d.ExportAsFixedFormat($pdf,17)
   Write-Output ($f.Name+' | pages='+$d.ComputeStatistics(2)+' | '+$pdf)
  } finally { if($null -ne $d){$d.Close(0)} }
 }
} finally { $app.Quit() }
